"""
메모리 효과를 고려한 Lorenz 96 모델 구현
- LSTM 기반 메모리 네트워크
- 과거 10개 시점을 활용한 동역학 예측
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from numba import jit, njit
from functools import partial
from tqdm import tqdm


# =============================================================================
# LSTM 메모리 네트워크 클래스
# =============================================================================
class LSTMMemoryNetwork(torch.nn.Module):
    """
    LSTM을 사용한 메모리 네트워크
    
    Args:
        input_dim (int): 입력 차원
        hidden_dim (int): LSTM 은닉 차원
        output_dim (int): 출력 차원
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=8):
        super(LSTMMemoryNetwork, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        순전파
        
        Args:
            x (torch.Tensor): 입력 시퀀스 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 출력 [batch_size, output_dim]
        """
        lstm_out, _ = self.lstm(x)  # LSTM 출력
        out = self.fc(lstm_out[:, -1, :])  # 마지막 시점의 출력만 사용
        return out

def stepper_2(Xt, model, F, sigma_X, mu_X, dt):
    """
    Xt: (batch_size, 2*n_hist+2, K)
    model: LSTMMemoryNetwork 인스턴스
    F: float
    sigma_X: (K,)
    mu_X: (K,)
    dt: float
    F: float

    RK4 방법으로 DDE를 적분하여 현재 시점의 state X를 계산
    """
    past_X2 = Xt[:,-6,:]   #Xt의 각 batch에서 마지막 6번째 시점의 값
    past_X = Xt[:,-4,:]    #Xt의 각 batch에서 마지막 4번째 시점의 값
    current_X = Xt[:,-2,:] #Xt의 각 batch에서 마지막 2번째 시점의 값

    Xdot1 = L96_2t_xdot_2(current_X, past_X, past_X2, model, F, sigma_X, mu_X)
    Xdot2 = L96_2t_xdot_2(current_X + 0.5 * dt * Xdot1, past_X, past_X2, model, F, sigma_X, mu_X)
    Xdot3 = L96_2t_xdot_2(current_X + 0.5 * dt * Xdot2, past_X, past_X2, model, F, sigma_X, mu_X)
    Xdot4 = L96_2t_xdot_2(current_X + dt * Xdot3, past_X, past_X2, model, F, sigma_X, mu_X)

    X_future = Xt[:,-2,:] + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))

    return X_future

def L96_2t_xdot_2(Xt, Xth1, Xth2, model, F, sigma_X, mu_X):
    """
    Xt: (batch_size, K) - PyTorch tensor
    Xth1: (batch_size, K) - PyTorch tensor
    Xth2: (batch_size, K) - PyTorch tensor
    """    
    # Lorenz 96 시스템의 기본 항
    Xdot = torch.roll(Xt, 1, dims=1) * (torch.roll(Xt, -1, dims=1) - torch.roll(Xt, 2, dims=1)) - Xt + F + forward_pass_2(Xt, Xth1, Xth2, model, sigma_X, mu_X)
    
    
    return Xdot

def forward_pass_2(Xt, Xth1, Xth2, model, sigma_X, mu_X):
    """
    LSTM을 사용한 forward pass 함수
    시계열 데이터를 과거→현재 순서로 구성하고
    정규화/역정규화 수행
    """
    Xt = Xt.float()
    Xth1 = Xth1.float()
    Xth2 = Xth2.float()
    
    # 정규화
    H = (Xt-mu_X)/sigma_X
    Hh1 = (Xth1-mu_X)/sigma_X
    Hh2 = (Xth2-mu_X)/sigma_X
    
    # LSTM 입력을 위한 시퀀스 구성
    sequence = torch.stack([Hh2, Hh1, H], dim=1)
    
    # LSTM forward pass
    output = model(sequence)
    
    # 역정규화
    output = output * sigma_X + mu_X
        
    return output

# =============================================================================
# 평가 및 외삽용 함수
# =============================================================================
def evaluate_forward_pass_2(Xt, Xth1, Xth2, model, sigma_X, mu_X):
    """
    LSTM을 사용한 forward pass 평가 함수
    
    Args:
        Xt (torch.Tensor/np.ndarray): 현재 시점의 상태
        Xth1 (torch.Tensor/np.ndarray): t-1 시점의 상태
        Xth2 (torch.Tensor/np.ndarray): t-2 시점의 상태
        model (LSTMMemoryNetwork): 학습된 LSTM 모델
        sigma_X (torch.Tensor/np.ndarray): 정규화를 위한 표준편차
        mu_X (torch.Tensor/np.ndarray): 정규화를 위한 평균
        
    Returns:
        torch.Tensor: 모델 예측값 (역정규화된 상태)
    """
    # numpy 입력을 tensor로 변환
    if isinstance(Xt, np.ndarray):
        Xt = torch.from_numpy(Xt.astype(np.float32))
        Xth1 = torch.from_numpy(Xth1.astype(np.float32))
        Xth2 = torch.from_numpy(Xth2.astype(np.float32))
        
    if isinstance(sigma_X, np.ndarray):
        sigma_X = torch.from_numpy(sigma_X.astype(np.float32))
        mu_X = torch.from_numpy(mu_X.astype(np.float32))
    
    # 평가 모드로 설정
    model.eval()
    
    with torch.no_grad():
        # 정규화
        H = (Xt - mu_X) / sigma_X
        Hh1 = (Xth1 - mu_X) / sigma_X
        Hh2 = (Xth2 - mu_X) / sigma_X
        
        # LSTM 입력을 위한 시퀀스 구성 (과거→현재 순서)
        sequence = torch.stack([Hh2, Hh1, H], dim=1)
        
        # LSTM forward pass
        output = model(sequence)
        
        # 역정규화
        output = output * sigma_X + mu_X
        
    return output


def evaluate_L96_2t_xdot_2(Xt, Xth1, Xth2, model, F, sigma_X, mu_X):    
    """
    LSTM 모델을 사용한 Lorenz 96 시스템의 시간 미분 계산
    """
    if isinstance(Xt, np.ndarray):
        Xt = torch.from_numpy(Xt.astype(np.float32))
        Xth1 = torch.from_numpy(Xth1.astype(np.float32))
        Xth2 = torch.from_numpy(Xth2.astype(np.float32))
    
    # Lorenz 96 시스템의 기본 항
    Xdot = torch.roll(Xt, 1, dims=1) * (torch.roll(Xt, -1, dims=1) - torch.roll(Xt, 2, dims=1)) - Xt + F
    
    # Memory term 추가
    memory_term = evaluate_forward_pass_2(Xt, Xth1, Xth2, model, sigma_X, mu_X)
    
    # 텐서 덧셈
    Xdot = Xdot + memory_term
    
    return Xdot  # numpy로 변환하지 않고 tensor 반환

def integrate_L96_2t_with_NN_2(X0, si, nt, model, F, sigma_X, mu_X, t0=0, dt=0.001):
    xhist = []
    X = torch.from_numpy(X0.copy().astype(np.float32))  # 초기부터 tensor로 변환
    xhist.append(X[0,:])
    for i in range(X.shape[0]-1):
        xhist.append(X[i+1,:])
    
    ns = 1
    for n in range(nt):
        if n%50 == 0:
            print(n,nt)
        for s in range(ns):
            # 현재 상태를 tensor로 유지
            x_current = xhist[-2][None,:]
            
            # RK4 update of X (모든 연산을 tensor로 수행)
            Xdot1 = evaluate_L96_2t_xdot_2(
                x_current, 
                xhist[-4][None,:], 
                xhist[-6][None,:], 
                model, F, sigma_X, mu_X)
            
            Xdot2 = evaluate_L96_2t_xdot_2(
                x_current + 0.5 * dt * Xdot1, 
                xhist[-3][None,:], 
                xhist[-5][None,:],
                model, F, sigma_X, mu_X)
            
            Xdot3 = evaluate_L96_2t_xdot_2(
                x_current + 0.5 * dt * Xdot2,
                xhist[-3][None,:], 
                xhist[-5][None,:],
                model, F, sigma_X, mu_X)
            
            Xdot4 = evaluate_L96_2t_xdot_2(
                x_current + dt * Xdot3,
                xhist[-2][None,:], 
                xhist[-4][None,:],
                model, F, sigma_X, mu_X)
            
            # 모든 연산을 tensor로 수행
            X = x_current + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
            
        xhist.append(X[0,:])
    
    # 마지막에 numpy로 변환하여 반환
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
    
    fold = os.path.join(os.getcwd(), 'memory_models_v2')  # 새로운 폴더명
    os.makedirs(fold, exist_ok=True)
    
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

    # Corrupting data with noise(PASS)
    np.save(os.path.join(fold, 'X_train.npy'), X_train)

    # First training routine where we target state at the next time-step
    n_hist = 2
    n_fut = 1

    Xt = []
    for i in range(2*n_hist+1+1):
        Xt.append(X_train[i:-2*n_hist-2+i-n_fut+1,:])
    Xt = np.transpose(np.array(Xt), (1, 0, 2)) # nt-2*n_hist-1 x 2*n_hist+2 x K
    Xtpdt = X_train[2*n_hist+2+n_fut-1:,:] # nt-2*n_hist-1 x K
    Ndata = Xt.shape[0]
    
    mu_X = np.zeros(X_train.shape[1])
    sigma_X = np.max(np.abs(X_train), axis=0)

    # LSTM 모델 초기화를 float32로 변경
    model = LSTMMemoryNetwork(K)  # float32로 변경
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    # 학습 데이터 준비 (float32로 변경)
    Xt = torch.from_numpy(Xt).float()
    Xtpdt = torch.from_numpy(Xtpdt).float()
    sigma_X = torch.from_numpy(sigma_X).float()
    mu_X = torch.from_numpy(mu_X).float()
    
    # 학습 파라미터 수정
    n_epochs = 1000  # 에폭 수 증가
    batch_size = 512
    
    # Early stopping 추가
    patience = 10
    best_loss = float('inf')
    patience_counter = 0
    
    # 학습 루프
    model.train()
    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0
        batch_count = 0
        
        for i in range(0, len(Xt), batch_size):
            batch_Xt = Xt[i:i+batch_size]
            batch_y = Xtpdt[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = stepper_2(batch_Xt, model, F, sigma_X, mu_X, dt)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
                    
        avg_epoch_loss = epoch_loss / batch_count
        print(f'Epoch {epoch}, Average Loss (float64): {avg_epoch_loss:.10f}')
        
    print(f'Training finished. Best loss: {best_loss:.6f}')

    # 모델 저장
    model_save_path = os.path.join(fold, 'lstm_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
        'sigma_X': sigma_X,
        'mu_X': mu_X,
        'hyperparameters': {
            'input_dim': K,
            'hidden_dim': 128,
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
        output_dim=checkpoint['hyperparameters']['output_dim']
    )
    eval_model.load_state_dict(checkpoint['model_state_dict'])
    eval_model.eval()

    nt = 1000
    X_int, Y_int, t, _ = integrate_L96_2t_with_coupling(X[0,:], Y[0,:], si/2, 2*nt, F, h, b, c, 0, dt/2)
    X_int = X_int[::2,:]
    Y_int = Y_int[::2,:]
    t = t[::2]

    X_int2dt = X_int[::2,:]
    Y_int2dt = Y_int[::2,:]
    t_2dt = t[::2]
    Xt_dt = X_int2dt[2:,:]
    Xth1_dt = X_int2dt[1:-1,:]
    Xth2_dt = X_int2dt[:-2,:]
    
    NN_out = evaluate_forward_pass_2(Xt_dt, Xth1_dt, Xth2_dt, eval_model, sigma_X, mu_X) 
    Xpred_int = integrate_L96_2t_with_NN_2(X_int[0:2*n_hist+2,:], si, nt-2*n_hist-1, eval_model, F, sigma_X, mu_X, 0, 2*dt)
    print(NN_out.shape)
    print(Xpred_int.shape)
    exact_out_int = []
    
    for ii in range(K):
        exact_out = - h*c/b*(Y_int2dt[2:,ii*J+0]+Y_int2dt[2:,ii*J+1]+Y_int2dt[2:,ii*J+2]+Y_int2dt[2:,ii*J+3]+
                             Y_int2dt[2:,ii*J+4]+Y_int2dt[2:,ii*J+5]+Y_int2dt[2:,ii*J+6]+Y_int2dt[2:,ii*J+7]+
                             Y_int2dt[2:,ii*J+8]+Y_int2dt[2:,ii*J+9]+Y_int2dt[2:,ii*J+10]+Y_int2dt[2:,ii*J+11]+
                             Y_int2dt[2:,ii*J+12]+Y_int2dt[2:,ii*J+13]+Y_int2dt[2:,ii*J+14]+Y_int2dt[2:,ii*J+15]+
                             Y_int2dt[2:,ii*J+16]+Y_int2dt[2:,ii*J+17]+Y_int2dt[2:,ii*J+18]+Y_int2dt[2:,ii*J+19]+
                             Y_int2dt[2:,ii*J+20]+Y_int2dt[2:,ii*J+21]+Y_int2dt[2:,ii*J+22]+Y_int2dt[2:,ii*J+23]+
                             Y_int2dt[2:,ii*J+24]+Y_int2dt[2:,ii*J+25]+Y_int2dt[2:,ii*J+26]+Y_int2dt[2:,ii*J+27]+
                             Y_int2dt[2:,ii*J+28]+Y_int2dt[2:,ii*J+29]+Y_int2dt[2:,ii*J+30]+Y_int2dt[2:,ii*J+31])   
        exact_out_int.append(exact_out)

    exact_out_int = np.array(exact_out_int)
    print(exact_out_int.shape)
    ####### Extrap ######
    Xpred_init = X[-2-2*n_hist:,:]
    
    X_ext, Y_ext, t_ext, _ = integrate_L96_2t_with_coupling(X[-1,:], Y[-1,:], si/2, 2*nt, F, h, b, c, 0, dt/2)
    X_ext = X_ext[::2,:]
    Y_ext = Y_ext[::2,:]
    t_ext = t_ext[::2]
    
    X_ext2dt = X_ext[::2,:]
    Y_ext2dt = Y_ext[::2,:]
    t_2dt_ext = t_ext[::2]
    Xt_dt = X_ext2dt[2:,:]
    Xth1_dt = X_ext2dt[1:-1,:]
    Xth2_dt = X_ext2dt[:-2,:]
    
    NN_out_ext = evaluate_forward_pass_2(Xt_dt, Xth1_dt, Xth2_dt, eval_model, sigma_X, mu_X) 
    print(NN_out_ext.shape)
    Xpred_ext = integrate_L96_2t_with_NN_2(Xpred_init, si, nt, eval_model, F, sigma_X, mu_X, 0, 2*dt)
    Xpred_ext = Xpred_ext[2*n_hist+1:,:]
    print(Xpred_ext.shape)
    exact_out_ext = []
    
    for ii in range(K):
        exact_out = - h*c/b*(Y_ext2dt[2:,ii*J+0]+Y_ext2dt[2:,ii*J+1]+Y_ext2dt[2:,ii*J+2]+Y_ext2dt[2:,ii*J+3]+
                             Y_ext2dt[2:,ii*J+4]+Y_ext2dt[2:,ii*J+5]+Y_ext2dt[2:,ii*J+6]+Y_ext2dt[2:,ii*J+7]+
                             Y_ext2dt[2:,ii*J+8]+Y_ext2dt[2:,ii*J+9]+Y_ext2dt[2:,ii*J+10]+Y_ext2dt[2:,ii*J+11]+
                             Y_ext2dt[2:,ii*J+12]+Y_ext2dt[2:,ii*J+13]+Y_ext2dt[2:,ii*J+14]+Y_ext2dt[2:,ii*J+15]+
                             Y_ext2dt[2:,ii*J+16]+Y_ext2dt[2:,ii*J+17]+Y_ext2dt[2:,ii*J+18]+Y_ext2dt[2:,ii*J+19]+
                             Y_ext2dt[2:,ii*J+20]+Y_ext2dt[2:,ii*J+21]+Y_ext2dt[2:,ii*J+22]+Y_ext2dt[2:,ii*J+23]+
                             Y_ext2dt[2:,ii*J+24]+Y_ext2dt[2:,ii*J+25]+Y_ext2dt[2:,ii*J+26]+Y_ext2dt[2:,ii*J+27]+
                             Y_ext2dt[2:,ii*J+28]+Y_ext2dt[2:,ii*J+29]+Y_ext2dt[2:,ii*J+30]+Y_ext2dt[2:,ii*J+31])
        exact_out_ext.append(exact_out)

    exact_out_ext = np.array(exact_out_ext)
    print(exact_out_ext.shape)
    err_int_det = np.linalg.norm(X_int-Xpred_int) / np.linalg.norm(X_int)
    err_ext_det = np.linalg.norm(X_ext-Xpred_ext) / np.linalg.norm(X_ext)
    print('Relative interpolation norm det: ',err_int_det)
    print('Relative extrapolation norm det: ',err_ext_det)
    
    err_int_NN_det = np.linalg.norm(exact_out_int-NN_out.T) / np.linalg.norm(exact_out_int)
    err_ext_NN_det = np.linalg.norm(exact_out_ext-NN_out_ext.T) / np.linalg.norm(exact_out_ext)
    print('Relative interpolation norm Closure det: ',err_int_NN_det)
    print('Relative extrapolation norm Closure det: ',err_ext_NN_det)

    np.save(fold+'/X_pred_int_det', Xpred_int)
    np.save(fold+'/X_int_det', X_int)
    np.save(fold+'/NN_int_det', NN_out)
    np.save(fold+'/exact_out_int_det', exact_out_int)
    np.save(fold+'/t_det', t)
    np.save(fold+'/t_2dt_det', t_2dt)
    
    np.save(fold+'/X_pred_ext_det', Xpred_ext)
    np.save(fold+'/X_ext_det', X_ext)
    np.save(fold+'/NN_ext_det', NN_out_ext)
    np.save(fold+'/exact_out_ext_det', exact_out_ext)
    np.save(fold+'/t_ext_det', t_ext)
    np.save(fold+'/t_2dt_ext_det', t_2dt_ext)
