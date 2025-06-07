"""
메모리 효과를 고려한 Lorenz 96 모델 구현
- LSTM 기반 메모리 네트워크
- 과거 10개 시점을 활용한 동역학 예측
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from numba import jit, njit
from functools import partial
from tqdm import tqdm


# =============================================================================
# LSTM 메모리 네트워크 클래스
# =============================================================================

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden_dim]
        scores = self.attention(hidden_states)  # [batch, seq_len, 1]
        weights = torch.softmax(scores, dim=1)  # [batch, seq_len, 1]
        context = torch.sum(hidden_states * weights, dim=1)  # [batch, hidden_dim]
        return context, weights

class LSTMMemoryNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=8, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # 입력 전처리 층
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 시간적 어텐션
        self.temporal_attention = TemporalAttention(hidden_dim)
        
        # 출력 레이어 (residual connection 포함)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),  # 원본 입력을 concat
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # 출력 안정화
        )
        
        # 출력 스케일 learnable parameter
        self.output_scale = nn.Parameter(torch.tensor(0.5))
        
        # 가중치 초기화
        self.reset_parameters()
        
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 입력 전처리
        x_proj = self.input_layer(x.reshape(-1, self.input_dim)).reshape(batch_size, seq_len, self.hidden_dim)
        
        # LSTM 통과
        lstm_out, _ = self.lstm(x_proj)
        
        # 시간적 어텐션 적용
        attended, _ = self.temporal_attention(lstm_out)
        
        # 원본 입력의 마지막 시점과 결합
        final_input = torch.cat([attended, x[:, -1]], dim=1)
        
        # 출력 예측
        output = self.output_layer(final_input)
        output = output * self.output_scale
        
        return output


def forward_pass_2(Xt, Xth1, Xth2, model, sigma_X, mu_X, is_eval=False):
    """
    LSTM을 사용한 forward pass 함수 (2개 과거 시점)
    - 정규화에 epsilon 추가
    - eval 모드 optional 지원
    """
    eps = 1e-6  # 작은 수로 안정성 확보

    # gradient 계산을 위해 requires_grad=True 설정
    Xt = Xt.float().requires_grad_(True)
    Xth1 = Xth1.float().requires_grad_(True)
    Xth2 = Xth2.float().requires_grad_(True)

    # 정규화 (과거 → 현재 순)
    Xt_norm = torch.clamp((Xt - mu_X) / (sigma_X + eps), -5.0, 5.0)
    Xth1_norm = torch.clamp((Xth1 - mu_X) / (sigma_X + eps), -5.0, 5.0)
    Xth2_norm = torch.clamp((Xth2 - mu_X) / (sigma_X + eps), -5.0, 5.0)

    sequence = torch.stack([Xth2_norm, Xth1_norm, Xt_norm], dim=1)

    # forward pass
    if is_eval:
        model.eval()
        with torch.no_grad():
            output = model(sequence)
    else:
        output = model(sequence)

    # 역정규화
    output = output * sigma_X + mu_X
    return output  # detach 제거


def L96_2t_xdot_2(Xt, Xth1, Xth2, model, F, sigma_X, mu_X, is_eval=False):
    """
    Lorenz 96 시스템의 시간 미분 계산 (memory term 포함)
    """
    # Lorenz 96 기본 항 계산
    core_term = torch.roll(Xt, 1, dims=1) * (torch.roll(Xt, -1, dims=1) - torch.roll(Xt, 2, dims=1)) - Xt + F

    # Memory term
    memory_term = forward_pass_2(Xt, Xth1, Xth2, model, sigma_X, mu_X, is_eval=is_eval)

    return core_term + memory_term


def stepper_2(Xt, model, F, sigma_X, mu_X, dt, is_eval=False):
    """
    RK4 방법으로 다음 시점 상태 계산
    Xt: [batch, 6, K] → 2개 과거 시점 기반 DDE
    """
    assert Xt.shape[1] == 6, "Xt should have shape [batch, 6, K] for 2-point memory"

    past_X2    = Xt[:, -6, :]  # t-2
    past_X1    = Xt[:, -4, :]  # t-1
    current_X  = Xt[:, -2, :]  # t

    # RK4 계산
    Xdot1 = L96_2t_xdot_2(current_X, past_X1, past_X2, model, F, sigma_X, mu_X, is_eval)
    Xdot2 = L96_2t_xdot_2(current_X + 0.5 * dt * Xdot1, past_X1, past_X2, model, F, sigma_X, mu_X, is_eval)
    Xdot3 = L96_2t_xdot_2(current_X + 0.5 * dt * Xdot2, past_X1, past_X2, model, F, sigma_X, mu_X, is_eval)
    Xdot4 = L96_2t_xdot_2(current_X + dt * Xdot3, past_X1, past_X2, model, F, sigma_X, mu_X, is_eval)

    X_future = current_X + (dt / 6.0) * (Xdot1 + Xdot4 + 2.0 * (Xdot2 + Xdot3))

    return X_future


def stepper_10(Xt, model, F, sigma_X, mu_X, dt, is_eval=False):
    """
    RK4 방법으로 다음 시점 상태 계산 (10개 시점 기반 DDE)
    - Xt: [batch, 22, K] → t-20 ~ t 까지 포함
    """
    assert Xt.shape[1] == 22, "Xt should have shape [batch, 22, K] for 10-point memory"

    current_X = Xt[:, -2, :]  # t 시점 값

    # RK4 계산
    Xdot1 = L96_2t_xdot_10(current_X, Xt, model, F, sigma_X, mu_X, is_eval)
    Xdot2 = L96_2t_xdot_10(current_X + 0.5 * dt * Xdot1, Xt, model, F, sigma_X, mu_X, is_eval)
    Xdot3 = L96_2t_xdot_10(current_X + 0.5 * dt * Xdot2, Xt, model, F, sigma_X, mu_X, is_eval)
    Xdot4 = L96_2t_xdot_10(current_X + dt * Xdot3, Xt, model, F, sigma_X, mu_X, is_eval)

    X_future = current_X + (dt / 6.0) * (Xdot1 + Xdot4 + 2.0 * (Xdot2 + Xdot3))
    return X_future

def L96_2t_xdot_10(Xt, X_seq, model, F, sigma_X, mu_X, is_eval=False):
    """
    Lorenz 96 시스템의 시간 미분 계산 (10개 시점 기반 memory term 사용)
    - Xt: [batch, K] (현재 시점)
    - X_seq: [batch, 22, K] (t-20 ~ t)
    """
    core_term = torch.roll(Xt, 1, dims=1) * (torch.roll(Xt, -1, dims=1) - torch.roll(Xt, 2, dims=1)) - Xt + F
    memory_term = forward_pass_10(X_seq, model, sigma_X, mu_X, is_eval=is_eval)

    return core_term + memory_term

def forward_pass_10(X_seq, model, sigma_X, mu_X, is_eval=False):
    """
    LSTM을 사용한 forward pass 함수 (10개 과거 시점)
    - X_seq: [batch_size, 2*n_hist+2=22, K]
    - 정규화 + 시퀀스 구성 + LSTM 예측
    """
    eps = 1e-6

    # gradient 계산을 위해 requires_grad=True 설정
    X_seq = X_seq.float().requires_grad_(True)
    
    # 최근 11개 시점 추출: t-10 ~ t
    recent_seq = X_seq[:, -11:, :]  # shape: [B, 11, K]
    
    # 정규화
    seq_norm = (recent_seq - mu_X) / (sigma_X + eps)
    seq_norm = torch.clamp(seq_norm, -5.0, 5.0)  # 안정화

    # 모델 예측
    if is_eval:
        model.eval()
        with torch.no_grad():
            output = model(seq_norm)
    else:
        output = model(seq_norm)

    # 역정규화
    output = output * sigma_X + mu_X
    return output  # detach 제거



def integrate_L96_2t_with_NN_2(X0, si, nt, model, F, sigma_X, mu_X, t0=0, dt=0.001):
    xhist = []
    X = torch.from_numpy(X0.astype(np.float32))  # 초기 입력 [2n_hist+2, K]
    for i in range(X.shape[0]):
        xhist.append(X[i])

    for _ in range(nt):
        Xt = torch.stack(xhist[-2:])          # t
        Xth1 = xhist[-4]                       # t-1
        Xth2 = xhist[-6]                       # t-2

        x_current = Xt[-1].unsqueeze(0)        # [1, K]

        # RK4 update
        Xdot1 = L96_2t_xdot_2(x_current, Xth1.unsqueeze(0), Xth2.unsqueeze(0), model, F, sigma_X, mu_X, is_eval=True)
        Xdot2 = L96_2t_xdot_2(x_current + 0.5 * dt * Xdot1, Xth1.unsqueeze(0), Xth2.unsqueeze(0), model, F, sigma_X, mu_X, is_eval=True)
        Xdot3 = L96_2t_xdot_2(x_current + 0.5 * dt * Xdot2, Xth1.unsqueeze(0), Xth2.unsqueeze(0), model, F, sigma_X, mu_X, is_eval=True)
        Xdot4 = L96_2t_xdot_2(x_current + dt * Xdot3, Xth1.unsqueeze(0), Xth2.unsqueeze(0), model, F, sigma_X, mu_X, is_eval=True)

        X_next = x_current + (dt / 6.0) * (Xdot1 + Xdot4 + 2.0 * (Xdot2 + Xdot3))
        xhist.append(X_next.squeeze(0))

    return torch.stack(xhist).detach().numpy()


def integrate_L96_2t_with_NN_10(X0, si, nt, model, F, sigma_X, mu_X, t0=0, dt=0.001):
    xhist = []
    X = torch.from_numpy(X0.astype(np.float32))  # 초기 입력 [2n_hist+2, K]
    for i in range(X.shape[0]):
        xhist.append(X[i])

    for _ in range(nt):
        X_seq = torch.stack(xhist[-22:]).unsqueeze(0)   # [1, 22, K]
        current_X = X_seq[:, -2, :]                     # [1, K]

        # RK4 update
        Xdot1 = L96_2t_xdot_10(current_X, X_seq, model, F, sigma_X, mu_X, is_eval=True)
        Xdot2 = L96_2t_xdot_10(current_X + 0.5 * dt * Xdot1, X_seq, model, F, sigma_X, mu_X, is_eval=True)
        Xdot3 = L96_2t_xdot_10(current_X + 0.5 * dt * Xdot2, X_seq, model, F, sigma_X, mu_X, is_eval=True)
        Xdot4 = L96_2t_xdot_10(current_X + dt * Xdot3, X_seq, model, F, sigma_X, mu_X, is_eval=True)

        X_next = current_X + (dt / 6.0) * (Xdot1 + Xdot4 + 2.0 * (Xdot2 + Xdot3))
        xhist.append(X_next.squeeze(0))

    return torch.stack(xhist).detach().numpy()


# =============================================================================
# 데이터 생성용 시간 적분 함수
# =============================================================================
@jit
def L96_2t_xdot_ydot(X, Y, F, h, b, c):
    JK, K = len(Y), len(X)
    J = JK // K
    assert JK == J * K, "X and Y have incompatible shapes"
    Xdot = np.zeros(K)
    hcb = (h * c) / b
    Ysummed = Y.reshape((K, J)).sum(axis=-1)
    Xdot = np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2)) - X + F - hcb * Ysummed
    Ydot = (
        -c * b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1))
        - c * Y
        + hcb * np.repeat(X, J)
    )
    return Xdot, Ydot, -hcb * Ysummed


def integrate_L96_2t_with_coupling(X0, Y0, si, nt, F, h, b, c, t0=0, dt=0.001):
    
    time, xhist, yhist, xytend_hist = [], [], [], []
    time.append(t0)
    
    X, Y = X0.copy(), Y0.copy()
    xhist.append(X)
    yhist.append(Y)
    if si < dt:
        dt, ns = si, 1
    else:
        ns = int(si / dt + 0.5)
        assert (
            abs(ns * dt - si) < 1e-14
        ), "si is not an integer multiple of dt: si=%f dt=%f ns=%i" % (si, dt, ns)
    for n in range(nt):
        if n%500 == 0:
            print(n,nt)
        for s in range(ns):
            # RK4 update of X,Y
            Xdot1, Ydot1, XYtend = L96_2t_xdot_ydot(X, Y, F, h, b, c)
            Xdot2, Ydot2, _ = L96_2t_xdot_ydot(
                X + 0.5 * dt * Xdot1, Y + 0.5 * dt * Ydot1, F, h, b, c
            )
            Xdot3, Ydot3, _ = L96_2t_xdot_ydot(
                X + 0.5 * dt * Xdot2, Y + 0.5 * dt * Ydot2, F, h, b, c
            )
            Xdot4, Ydot4, _ = L96_2t_xdot_ydot(
                X + dt * Xdot3, Y + dt * Ydot3, F, h, b, c
            )
            X = X + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
            Y = Y + (dt / 6.0) * ((Ydot1 + Ydot4) + 2.0 * (Ydot2 + Ydot3))

        xhist.append(X)
        yhist.append(Y)
        time.append(t0 + si * (n + 1))
        xytend_hist.append(XYtend)

    return np.array(xhist), np.array(yhist), np.array(time), np.array(xytend_hist)


def s(k, K):
    """A non-dimension coordinate from -1..+1 corresponding to k=0..K"""
    return 2*k/K - 1


if __name__ == "__main__":
    print("메모리 효과를 고려한 Lorenz 96 모델")
    
    # 과거 시점 개수 선택 (2 또는 10)
    use_memory_points = 10  # 여기서 2나 10으로 설정
    
    fold = os.path.join(os.getcwd(), f'memory_models_v2_{use_memory_points}points')  # 폴더명에 시점 수 추가
    os.makedirs(fold, exist_ok=True)
    model_save_path = os.path.join(fold, 'lstm_model.pth')

    # 시스템 파라미터 설정
    K = 8  # Number of globa-scale variables X
    J = 32  # Number of local-scale Y variables per single global-scale X variable
    F = 15.0  # Forcing
    b = 10.0  # ratio of amplitudes
    c = 10.0  # time-scale ratio
    h = 1.0  # Coupling coefficient
        
    # 평가 파라미터 설정
    nt_pre = 20000  # Number of time steps for model spinup
    nt = 20000  # Number of time steps
    si = 0.005  # Sampling time interval
    dt = 0.005  # Time step
    dt = dt*2
    si = si*2
    
    noise = 0.03

    # Initial conditions
    k = np.arange(K)
    j = np.arange(J * K)
    Xinit = s(k, K) * (s(k, K) - 1) * (s(k, K) + 1)
    Yinit = 0 * s(j, J * K) * (s(j, J * K) - 1) * (s(j, J * K) + 1)
    
    # Solving true model
    X, Y, t2, _ = integrate_L96_2t_with_coupling(Xinit, Yinit, si, nt_pre+nt, F, h, b, c, dt=dt)
    X = X[nt_pre:,:]
    Y = Y[nt_pre:,:]

    # Sub-sampling (tmeporal sparsity)
    X_train = X[::2,:]

    # First training routine where we target state at the next time-step
    if use_memory_points == 2:
        n_hist = 2  # 2개의 과거 시점
    else:
        n_hist = 10  # 10개의 과거 시점
    n_fut = 1
    # Corrupting data with noise(PASS)
    np.save(os.path.join(fold, 'X_train.npy'), X_train)

    Xt = []
    for i in range(2*n_hist+1+1):
        Xt.append(X_train[i:-2*n_hist-2+i-n_fut+1,:])
    Xt = np.transpose(np.array(Xt), (1, 0, 2)) # nt-2*n_hist-1 x 2*n_hist+2 x K
    Xtpdt = X_train[2*n_hist+2+n_fut-1:,:] # nt-2*n_hist-1 x K
    Ndata = Xt.shape[0]
        
    mu_X = np.mean(X_train, axis=0)
    sigma_X = np.std(X_train, axis=0) + 1e-6  # 안정성 보장용 epsilon
    # 학습 데이터 준비 (float32로 변경)
    Xt = torch.from_numpy(Xt).float()
    Xtpdt = torch.from_numpy(Xtpdt).float()
    sigma_X = torch.from_numpy(sigma_X).float()
    mu_X = torch.from_numpy(mu_X).float()
    
    # LSTM 모델 초기화를 float32로 변경
    model = LSTMMemoryNetwork(
        input_dim=8,      # Lorenz 96의 K값
        hidden_dim=256,   # 은닉 차원
        output_dim=8,     # 출력 차원 (K와 동일)
        num_layers=2,     # LSTM 층 수
        dropout=0.3      # 드롭아웃 비율
    )
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()    
    
    # 학습 파라미터 수정
    n_epochs = 100  # 에폭 수 증가
    batch_size = 512
        
    # 학습 루프
    model.train()
    losses = []
    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0
        batch_count = 0
        
        for i in range(0, len(Xt), batch_size):
            batch_Xt = Xt[i:i+batch_size]
            batch_y = Xtpdt[i:i+batch_size]
            
            optimizer.zero_grad()
            if use_memory_points == 2:
                pred = stepper_2(batch_Xt, model, F, sigma_X, mu_X, dt)
            else:  # use_memory_points == 10
                pred = stepper_10(batch_Xt, model, F, sigma_X, mu_X, dt)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
                    
        avg_epoch_loss = epoch_loss / batch_count
        print(f'Epoch {epoch}, Average Loss: {avg_epoch_loss:.10f}')
        losses.append(avg_epoch_loss)
        
    print(f'Training finished')

    # 손실값 플롯 저장
    plt.figure(figsize=(10, 6))
    plt.plot(range(n_epochs), losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fold, 'training_loss.png'))
    plt.close()

    # 모델 저장
    model_save_path = os.path.join(fold, 'lstm_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'sigma_X': sigma_X,
        'mu_X': mu_X,
        'hyperparameters': {
            'input_dim': K,
            'hidden_dim': 256,
            'output_dim': K,
            'batch_size': batch_size,
            'learning_rate': 1e-3
        }
    }, model_save_path)
    print(f'모델이 저장되었습니다: {model_save_path}')

    # Evaluation
    # 평가를 위한 모델 로드
    checkpoint = torch.load(model_save_path)
    eval_model = LSTMMemoryNetwork(
        input_dim=checkpoint['hyperparameters']['input_dim'],
        hidden_dim=checkpoint['hyperparameters']['hidden_dim'],
        output_dim=checkpoint['hyperparameters']['output_dim'],
        num_layers=2  # 학습 시와 동일하게 2개의 레이어 설정
    )
    eval_model.load_state_dict(checkpoint['model_state_dict'])
    eval_model.eval()

    # Interpolation 평가
    # 시뮬레이션 시간 스텝 수
    nt = 1000

    # 시뮬레이션 실행
    X_int, Y_int, t, _ = integrate_L96_2t_with_coupling(X[0,:], Y[0,:], si/2, 2*nt, F, h, b, c, 0, dt/2)
    print(X_int.shape, Y_int.shape, t.shape)

    # 2칸씩 건너뛰면서 서브샘플링
    X_int = X_int[::2,:]
    Y_int = Y_int[::2,:]
    t = t[::2]
    print(X_int.shape, Y_int.shape, t.shape)

    # 2칸씩 건너뛰면서 서브샘플링
    X_int2dt = X_int[::2,:]
    Y_int2dt = Y_int[::2,:]
    t_2dt = t[::2]
    print(X_int2dt.shape, Y_int2dt.shape, t_2dt.shape)

    # 시간 시프트된 데이터 준비 (t, t-1, t-2 시점의 데이터)
    if use_memory_points == 2:
        # 2개 과거 시점용 데이터 준비
        Xt_dt = X_int2dt[2:,:]
        Xth1_dt = X_int2dt[1:-1,:]
        Xth2_dt = X_int2dt[:-2,:]
        
        # 모델 출력 계산
        NN_out = forward_pass_2(Xt_dt, Xth1_dt, Xth2_dt, eval_model, sigma_X, mu_X)
        Xpred_int = integrate_L96_2t_with_NN_2(X_int[0:2*n_hist+2,:], si, nt-2*n_hist-1, eval_model, F, sigma_X, mu_X, 0, 2*dt)
    else:
        # 10개 과거 시점용 데이터 준비
        Xt_dt = X_int2dt[10:,:]
        Xth1_dt = X_int2dt[9:-1,:]
        Xth2_dt = X_int2dt[8:-2,:]
        Xth3_dt = X_int2dt[7:-3,:]
        Xth4_dt = X_int2dt[6:-4,:]
        Xth5_dt = X_int2dt[5:-5,:]
        Xth6_dt = X_int2dt[4:-6,:]
        Xth7_dt = X_int2dt[3:-7,:]
        Xth8_dt = X_int2dt[2:-8,:]
        Xth9_dt = X_int2dt[1:-9,:]
        Xth10_dt = X_int2dt[:-10,:]
        
        # 10개 과거 시점용 데이터 준비
        # numpy 배열을 tensor로 변환
        Xt_dt = torch.from_numpy(Xt_dt).float()
        Xth1_dt = torch.from_numpy(Xth1_dt).float()
        Xth2_dt = torch.from_numpy(Xth2_dt).float()
        Xth3_dt = torch.from_numpy(Xth3_dt).float()
        Xth4_dt = torch.from_numpy(Xth4_dt).float()
        Xth5_dt = torch.from_numpy(Xth5_dt).float()
        Xth6_dt = torch.from_numpy(Xth6_dt).float()
        Xth7_dt = torch.from_numpy(Xth7_dt).float()
        Xth8_dt = torch.from_numpy(Xth8_dt).float()
        Xth9_dt = torch.from_numpy(Xth9_dt).float()
        Xth10_dt = torch.from_numpy(Xth10_dt).float()
        
        X_seq = torch.stack([
            Xth10_dt, Xth9_dt, Xth8_dt, Xth7_dt, Xth6_dt,
            Xth5_dt, Xth4_dt, Xth3_dt, Xth2_dt, Xth1_dt,
            Xt_dt
        ], dim=1)  # [N, 11, K] 형태로 변환
        
        # 모델 출력 계산
        NN_out = forward_pass_10(X_seq, eval_model, sigma_X, mu_X)
        Xpred_int = integrate_L96_2t_with_NN_10(X_int[0:2*n_hist+2,:], si, nt-2*n_hist-1, eval_model, F, sigma_X, mu_X, 0, 2*dt)

    print("NN_out.shape", NN_out.shape)
    print("Xpred_int.shape", Xpred_int.shape)
    exact_out_int = []
    
    if use_memory_points == 2:
        start_idx = 2  # 2개의 과거 시점 사용
    else:  # use_memory_points == 10
        start_idx = 10  # 10개의 과거 시점 사용

    for ii in range(K):
        exact_out = - h*c/b*(Y_int2dt[start_idx:,ii*J+0]+Y_int2dt[start_idx:,ii*J+1]+Y_int2dt[start_idx:,ii*J+2]+Y_int2dt[start_idx:,ii*J+3]+
                             Y_int2dt[start_idx:,ii*J+4]+Y_int2dt[start_idx:,ii*J+5]+Y_int2dt[start_idx:,ii*J+6]+Y_int2dt[start_idx:,ii*J+7]+
                             Y_int2dt[start_idx:,ii*J+8]+Y_int2dt[start_idx:,ii*J+9]+Y_int2dt[start_idx:,ii*J+10]+Y_int2dt[start_idx:,ii*J+11]+
                             Y_int2dt[start_idx:,ii*J+12]+Y_int2dt[start_idx:,ii*J+13]+Y_int2dt[start_idx:,ii*J+14]+Y_int2dt[start_idx:,ii*J+15]+
                             Y_int2dt[start_idx:,ii*J+16]+Y_int2dt[start_idx:,ii*J+17]+Y_int2dt[start_idx:,ii*J+18]+Y_int2dt[start_idx:,ii*J+19]+
                             Y_int2dt[start_idx:,ii*J+20]+Y_int2dt[start_idx:,ii*J+21]+Y_int2dt[start_idx:,ii*J+22]+Y_int2dt[start_idx:,ii*J+23]+
                             Y_int2dt[start_idx:,ii*J+24]+Y_int2dt[start_idx:,ii*J+25]+Y_int2dt[start_idx:,ii*J+26]+Y_int2dt[start_idx:,ii*J+27]+
                             Y_int2dt[start_idx:,ii*J+28]+Y_int2dt[start_idx:,ii*J+29]+Y_int2dt[start_idx:,ii*J+30]+Y_int2dt[start_idx:,ii*J+31])   
        exact_out_int.append(exact_out)

    exact_out_int = np.array(exact_out_int)
    print("exact_out_int.shape", exact_out_int.shape)

    ####### Extrap ######
    Xpred_init = X[-2-2*n_hist:,:]
    
    X_ext, Y_ext, t_ext, _ = integrate_L96_2t_with_coupling(X[-1,:], Y[-1,:], si/2, 2*nt, F, h, b, c, 0, dt/2)
    X_ext = X_ext[::2,:]
    Y_ext = Y_ext[::2,:]
    t_ext = t_ext[::2]
    
    X_ext2dt = X_ext[::2,:]
    Y_ext2dt = Y_ext[::2,:]
    t_2dt_ext = t_ext[::2]

    if use_memory_points == 2:
        # 2개 과거 시점용 데이터 준비
        Xt_dt = X_ext2dt[2:,:]
        Xth1_dt = X_ext2dt[1:-1,:]
        Xth2_dt = X_ext2dt[:-2,:]
        
        # 모델 출력 계산
        NN_out_ext = forward_pass_2(Xt_dt, Xth1_dt, Xth2_dt, eval_model, sigma_X, mu_X)
        Xpred_ext = integrate_L96_2t_with_NN_2(Xpred_init, si, nt, eval_model, F, sigma_X, mu_X, 0, 2*dt)
    else:
        # 10개 과거 시점용 데이터 준비
        Xt_dt = X_ext2dt[10:,:]
        Xth1_dt = X_ext2dt[9:-1,:]
        Xth2_dt = X_ext2dt[8:-2,:]
        Xth3_dt = X_ext2dt[7:-3,:]
        Xth4_dt = X_ext2dt[6:-4,:]
        Xth5_dt = X_ext2dt[5:-5,:]
        Xth6_dt = X_ext2dt[4:-6,:]
        Xth7_dt = X_ext2dt[3:-7,:]
        Xth8_dt = X_ext2dt[2:-8,:]
        Xth9_dt = X_ext2dt[1:-9,:]
        Xth10_dt = X_ext2dt[:-10,:]
        
        # 10개 과거 시점용 데이터 준비
        # numpy 배열을 tensor로 변환
        Xt_dt = torch.from_numpy(Xt_dt).float()
        Xth1_dt = torch.from_numpy(Xth1_dt).float()
        Xth2_dt = torch.from_numpy(Xth2_dt).float()
        Xth3_dt = torch.from_numpy(Xth3_dt).float()
        Xth4_dt = torch.from_numpy(Xth4_dt).float()
        Xth5_dt = torch.from_numpy(Xth5_dt).float()
        Xth6_dt = torch.from_numpy(Xth6_dt).float()
        Xth7_dt = torch.from_numpy(Xth7_dt).float()
        Xth8_dt = torch.from_numpy(Xth8_dt).float()
        Xth9_dt = torch.from_numpy(Xth9_dt).float()
        Xth10_dt = torch.from_numpy(Xth10_dt).float()
        
        X_seq = torch.stack([
            Xth10_dt, Xth9_dt, Xth8_dt, Xth7_dt, Xth6_dt,
            Xth5_dt, Xth4_dt, Xth3_dt, Xth2_dt, Xth1_dt,
            Xt_dt
        ], dim=1)  # [N, 11, K] 형태로 변환
        
        # 모델 출력 계산
        NN_out_ext = forward_pass_10(X_seq, eval_model, sigma_X, mu_X)
        Xpred_ext = integrate_L96_2t_with_NN_10(Xpred_init, si, nt, eval_model, F, sigma_X, mu_X, 0, 2*dt)

    print("NN_out_ext.shape", NN_out_ext.shape)
    Xpred_ext = Xpred_ext[2*n_hist+1:,:]
    print("Xpred_ext.shape", Xpred_ext.shape)
    
    exact_out_ext = []
    for ii in range(K):
        exact_out = - h*c/b*(Y_ext2dt[start_idx:,ii*J+0]+Y_ext2dt[start_idx:,ii*J+1]+Y_ext2dt[start_idx:,ii*J+2]+Y_ext2dt[start_idx:,ii*J+3]+
                             Y_ext2dt[start_idx:,ii*J+4]+Y_ext2dt[start_idx:,ii*J+5]+Y_ext2dt[start_idx:,ii*J+6]+Y_ext2dt[start_idx:,ii*J+7]+
                             Y_ext2dt[start_idx:,ii*J+8]+Y_ext2dt[start_idx:,ii*J+9]+Y_ext2dt[start_idx:,ii*J+10]+Y_ext2dt[start_idx:,ii*J+11]+
                             Y_ext2dt[start_idx:,ii*J+12]+Y_ext2dt[start_idx:,ii*J+13]+Y_ext2dt[start_idx:,ii*J+14]+Y_ext2dt[start_idx:,ii*J+15]+
                             Y_ext2dt[start_idx:,ii*J+16]+Y_ext2dt[start_idx:,ii*J+17]+Y_ext2dt[start_idx:,ii*J+18]+Y_ext2dt[start_idx:,ii*J+19]+
                             Y_ext2dt[start_idx:,ii*J+20]+Y_ext2dt[start_idx:,ii*J+21]+Y_ext2dt[start_idx:,ii*J+22]+Y_ext2dt[start_idx:,ii*J+23]+
                             Y_ext2dt[start_idx:,ii*J+24]+Y_ext2dt[start_idx:,ii*J+25]+Y_ext2dt[start_idx:,ii*J+26]+Y_ext2dt[start_idx:,ii*J+27]+
                             Y_ext2dt[start_idx:,ii*J+28]+Y_ext2dt[start_idx:,ii*J+29]+Y_ext2dt[start_idx:,ii*J+30]+Y_ext2dt[start_idx:,ii*J+31])
        exact_out_ext.append(exact_out)

    exact_out_ext = np.array(exact_out_ext)
    print("exact_out_ext.shape", exact_out_ext.shape)
    err_int_det = np.linalg.norm(X_int-Xpred_int) / np.linalg.norm(X_int)
    err_int_NN_det = np.linalg.norm(exact_out_int-NN_out.detach().numpy().T) / np.linalg.norm(exact_out_int)
    print('Relative interpolation norm det: ',err_int_det)
    print('Relative interpolation norm Closure det: ',err_int_NN_det)
    
    err_ext_det = np.linalg.norm(X_ext-Xpred_ext) / np.linalg.norm(X_ext)
    err_ext_NN_det = np.linalg.norm(exact_out_ext-NN_out_ext.detach().numpy().T) / np.linalg.norm(exact_out_ext)
    print('Relative extrapolation norm det: ',err_ext_det)
    print('Relative extrapolation norm Closure det: ',err_ext_NN_det)

    # 결과 저장
    np.save(fold+'/X_pred_int_det', Xpred_int)
    np.save(fold+'/X_int_det', X_int)
    np.save(fold+'/NN_int_det', NN_out.detach().numpy())
    np.save(fold+'/exact_out_int_det', exact_out_int)
    np.save(fold+'/t_det', t)
    np.save(fold+'/t_2dt_det', t_2dt)
    
    np.save(fold+'/X_pred_ext_det', Xpred_ext)
    np.save(fold+'/X_ext_det', X_ext)
    np.save(fold+'/NN_ext_det', NN_out_ext.detach().numpy())
    np.save(fold+'/exact_out_ext_det', exact_out_ext)
    np.save(fold+'/t_ext_det', t_ext)
    np.save(fold+'/t_2dt_ext_det', t_2dt_ext)
    