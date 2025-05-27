"""
메모리 효과를 고려한 Lorenz 96 모델 구현
- LSTM 기반 메모리 네트워크
- 과거 10개 시점을 활용한 동역학 예측
"""

import os
import torch
import jax.numpy as np
from jax import jit, random, grad, vmap
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
    def __init__(self, input_dim, hidden_dim, output_dim):
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


# =============================================================================
# 10개 과거 시점을 사용하는 신경망 순전파 함수
# =============================================================================
def forward_pass_10(Hor, Horh1, Horh2, Horh3, Horh4, Horh5, Horh6, Horh7, Horh8, Horh9, Horh10, 
                   W_in, num_par_NN, K, L, sigma_X, mu_X):
    """
    10개의 과거 시점을 고려하는 신경망 순전파
    
    Args:
        Hor: 현재 시점 상태
        Horh1~Horh10: 과거 1~10 시점 상태들
        W_in: 신경망 가중치
        num_par_NN: 신경망 파라미터 수
        K: 상태 변수 개수
        L: 각 층의 뉴런 수 리스트
        sigma_X, mu_X: 정규화 파라미터
        
    Returns:
        numpy.ndarray: 신경망 출력
    """
    num_layers = len(L)
    
    # 입력 정규화
    H = (Hor - mu_X) / sigma_X
    Hh1 = (Horh1 - mu_X) / sigma_X
    Hh2 = (Horh2 - mu_X) / sigma_X
    Hh3 = (Horh3 - mu_X) / sigma_X
    Hh4 = (Horh4 - mu_X) / sigma_X
    Hh5 = (Horh5 - mu_X) / sigma_X
    Hh6 = (Horh6 - mu_X) / sigma_X
    Hh7 = (Horh7 - mu_X) / sigma_X
    Hh8 = (Horh8 - mu_X) / sigma_X
    Hh9 = (Horh9 - mu_X) / sigma_X
    Hh10 = (Horh10 - mu_X) / sigma_X
    
    # 첫 번째 변수에 대한 처리
    Ho = np.concatenate((H[:,0:1], Hh1[:,0:1], Hh2[:,0:1], Hh3[:,0:1], Hh4[:,0:1], Hh5[:,0:1],
                         Hh6[:,0:1], Hh7[:,0:1], Hh8[:,0:1], Hh9[:,0:1], Hh10[:,0:1]), axis=1)
    
    # 첫 번째 변수에 대한 신경망 계산
    Wl = W_in[:num_par_NN]
    
    # 은닉층들 처리
    for k in range(0, num_layers-2):
        W = np.reshape(Wl[0:L[k] * L[k+1]], (L[k], L[k+1]))
        Wl = Wl[L[k] * L[k+1]:]
        b = np.reshape(Wl[0:L[k+1]], (1, L[k+1]))
        Wl = Wl[L[k+1]:]
        Ho = np.tanh(np.add(np.matmul(Ho, W), b))
    
    # 출력층 처리
    W = np.reshape(Wl[0:L[num_layers-2] * L[num_layers-1]], (L[num_layers-2], L[num_layers-1]))
    Wl = Wl[L[num_layers-2] * L[num_layers-1]:]
    b = np.reshape(Wl[0:L[num_layers-1]], (1, L[num_layers-1]))
    Ho = np.add(np.matmul(Ho, W), b)
    
    # 나머지 K-1개 변수들에 대한 처리
    for kk in range(K-1):
        # kk+1번째 변수에 대한 입력 구성
        Hl = np.concatenate((H[:,kk+1:kk+2], Hh1[:,kk+1:kk+2], Hh2[:,kk+1:kk+2], Hh3[:,kk+1:kk+2], 
                             Hh4[:,kk+1:kk+2], Hh5[:,kk+1:kk+2], Hh6[:,kk+1:kk+2], Hh7[:,kk+1:kk+2], 
                             Hh8[:,kk+1:kk+2], Hh9[:,kk+1:kk+2], Hh10[:,kk+1:kk+2]), axis=1)
        
        # kk+1번째 변수에 대한 가중치 추출
        Wl = W_in[(kk+1)*num_par_NN:(kk+2)*num_par_NN]
        
        # 은닉층들 처리
        for k in range(0, num_layers-2):
            W = np.reshape(Wl[0:L[k] * L[k+1]], (L[k], L[k+1]))
            Wl = Wl[L[k] * L[k+1]:]
            b = np.reshape(Wl[0:L[k+1]], (1, L[k+1]))
            Wl = Wl[L[k+1]:]
            Hl = np.tanh(np.add(np.matmul(Hl, W), b))
        
        # 출력층 처리
        W = np.reshape(Wl[0:L[num_layers-2] * L[num_layers-1]], (L[num_layers-2], L[num_layers-1]))
        Wl = Wl[L[num_layers-2] * L[num_layers-1]:]
        b = np.reshape(Wl[0:L[num_layers-1]], (1, L[num_layers-1]))
        Hl = np.add(np.matmul(Hl, W), b)
        
        # 결과 연결
        Ho = np.concatenate((Ho, Hl), axis=1)
    
    return Ho


# =============================================================================
# 메모리 효과를 포함한 Lorenz 96 미분방정식
# =============================================================================
def L96_2t_xdot_10(Xt, Xth1, Xth2, Xth3, Xth4, Xth5, Xth6, Xth7, Xth8, Xth9, Xth10, 
                   W, num_par_NN, K, L, sigma_X, mu_X, F):
    """
    10개 과거 시점을 고려한 Lorenz 96 미분방정식
    
    Args:
        Xt: 현재 시점 상태
        Xth1~Xth10: 과거 1~10 시점 상태들
        W: 신경망 가중치
        num_par_NN: 신경망 파라미터 수
        K: 상태 변수 개수
        L: 각 층의 뉴런 수 리스트
        sigma_X, mu_X: 정규화 파라미터
        F: 외부 강제력
        
    Returns:
        numpy.ndarray: 시간 미분값
    """
    # Lorenz 96 기본 동역학 + 신경망 메모리 항
    Xdot = (np.roll(Xt, 1, axis=1) * (np.roll(Xt, -1, axis=1) - np.roll(Xt, 2, axis=1)) - 
            Xt + F + 
            forward_pass_10(Xt, Xth1, Xth2, Xth3, Xth4, Xth5, Xth6, Xth7, Xth8, Xth9, Xth10, 
                           W, num_par_NN, K, L, sigma_X, mu_X))
    
    return Xdot


# =============================================================================
# 메모리 효과를 포함한 시간 적분 함수
# =============================================================================

def integrate_L96_2t_with_NN_10(X0, si, nt, params, model, F, t0=0, dt=0.001):
    """
    10개 과거 시점을 고려한 Lorenz 96 시스템 시간 적분
    
    Args:
        X0: 초기 조건 (과거 시점들 포함)
        si: 시간 간격
        nt: 시간 스텝 수
        params: 모델 파라미터
        model: 신경망 모델
        F: 외부 강제력
        t0: 시작 시간
        dt: 적분 시간 간격
        
    Returns:
        numpy.ndarray: 시간 진화 결과
    """
    xhist = []
    X = X0.copy()
    
    # 초기 조건들을 히스토리에 추가
    for i in range(X.shape[0]):
        xhist.append(X[i, :])
    
    ns = 1  # 서브스텝 수
    
    # 시간 적분 루프
    for n in range(nt):
        if n % 50 == 0:
            print(f"Progress: {n}/{nt}")
        
        for s in range(ns):
            # RK4 적분 방법
            # 1단계
            Xdot1 = L96_2t_xdot_10(xhist[-2][None,:], xhist[-4][None,:], xhist[-6][None,:], 
                                   xhist[-8][None,:], xhist[-10][None,:], xhist[-12][None,:], 
                                   xhist[-14][None,:], xhist[-16][None,:], xhist[-18][None,:], 
                                   xhist[-20][None,:], xhist[-22][None,:], 
                                   model.params, model.num_par_NN, model.K, model.L, 
                                   model.sigma_X, model.mu_X, F)
            
            # 2단계
            Xdot2 = L96_2t_xdot_10(xhist[-2][None,:] + 0.5 * dt * Xdot1, xhist[-3][None,:], 
                                   xhist[-5][None,:], xhist[-7][None,:], xhist[-9][None,:], 
                                   xhist[-11][None,:], xhist[-13][None,:], xhist[-15][None,:], 
                                   xhist[-17][None,:], xhist[-19][None,:], xhist[-21][None,:], 
                                   model.params, model.num_par_NN, model.K, model.L, 
                                   model.sigma_X, model.mu_X, F)
            
            # 3단계
            Xdot3 = L96_2t_xdot_10(xhist[-2][None,:] + 0.5 * dt * Xdot2, xhist[-3][None,:], 
                                   xhist[-5][None,:], xhist[-7][None,:], xhist[-9][None,:], 
                                   xhist[-11][None,:], xhist[-13][None,:], xhist[-15][None,:], 
                                   xhist[-17][None,:], xhist[-19][None,:], xhist[-21][None,:], 
                                   model.params, model.num_par_NN, model.K, model.L, 
                                   model.sigma_X, model.mu_X, F)
            
            # 4단계
            Xdot4 = L96_2t_xdot_10(xhist[-2][None,:] + dt * Xdot3, xhist[-2][None,:], 
                                   xhist[-4][None,:], xhist[-6][None,:], xhist[-8][None,:], 
                                   xhist[-10][None,:], xhist[-12][None,:], xhist[-14][None,:], 
                                   xhist[-16][None,:], xhist[-18][None,:], xhist[-20][None,:], 
                                   model.params, model.num_par_NN, model.K, model.L, 
                                   model.sigma_X, model.mu_X, F)
            
            # RK4 업데이트
            X = xhist[-2][None,:] + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
        
        # 새로운 상태를 히스토리에 추가
        xhist.append(X[0,:])
        
    return np.array(xhist)


# =============================================================================
# 메인 실행부
# =============================================================================
import jax.numpy as np
from jax import jit

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
    fold = os.path.join(os.getcwd(), 'memory_models')
    os.makedirs(fold, exist_ok=True)
    
    K = 8 # Number of globa-scale variables X
    J = 32 # Number of local-scale Y variables per single global-scale X variable
    F = 15.0 # Focring
    b = 10.0 # ratio of amplitudes
    c = 10.0 # time-scale ratio
    h = 1.0 # Coupling coefficient
    noise = 0.03
    n_hist = 10
    nt_pre = 20000 # Number of time steps for model spinup
    nt = 20000  # Number of time steps
    si = 0.005  # Sampling time interval
    dt = 0.005  # Time step

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
    X = X[::2,:]
    dt = dt*2
    si = si*2

    # Corrupting data with noise
    X_train  = X + noise*X.std(0)*random.normal(random.PRNGKey(1234), X.shape)  
    np.save(os.path.join(fold, 'X_train.npy'), X_train)

    # First training routine where we target state at the next time-step
    n_fut = 1

    # Prepare start and end points given n_hist and n_fut
    Xt = []
    for i in range(2*n_hist+1+1):
        Xt.append(X_train[i:-2*n_hist-2+i-n_fut+1,:])
    Xt = np.transpose(np.array(Xt), (1, 0, 2)) # nt-2*n_hist-1 x 2*n_hist+2 x K
    Xtpdt = X_train[2*n_hist+2+n_fut-1:,:] # nt-2*n_hist-1 x K
    Ndata = Xt.shape[0]