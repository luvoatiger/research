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
# CNN-LSTM 메모리 네트워크 클래스
# =============================================================================
import torch
import torch.nn as nn

class CNNLSTMMemoryNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=8, num_layers=1, dropout=0.3):
        super(CNNLSTMMemoryNetwork, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # CNN 레이어 (시간적 특징 추출)
        self.conv_layers = nn.Sequential(
            # 첫 번째 CNN 레이어
            nn.Conv1d(input_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.LeakyReLU(),
            nn.Dropout(dropout/2),
            
            # 두 번째 CNN 레이어
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout/2)
        )

        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # 인과성 유지
        )

        # 특징 정규화
        self.feature_norm = nn.LayerNorm(hidden_dim)

        # 출력 레이어 (잔차 연결 포함)
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim//2),  # 잔차 연결을 위해 input_dim 추가
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, output_dim),
            nn.Tanh()
        )

        # 출력 스케일 학습 파라미터
        self.output_scale = nn.Parameter(torch.tensor(0.5))

        # 가중치 초기화
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    # CNN과 Linear 레이어용 초기화
                    nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')
                else:
                    # 1D 가중치 초기화
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # CNN 입력을 위한 형태 변환 [batch, input_dim, seq_len]
        x_conv = x.transpose(1, 2)
        
        # CNN 특징 추출
        conv_features = self.conv_layers(x_conv)
        
        # LSTM 입력을 위한 형태 변환 [batch, seq_len, hidden_dim]
        lstm_in = conv_features.transpose(1, 2)
        
        # LSTM 처리
        lstm_out, _ = self.lstm(lstm_in)
        
        # 마지막 시점 출력
        last_output = lstm_out[:, -1, :]
        
        # 정규화
        normed = self.feature_norm(last_output)
        
        # 잔차 연결을 위한 마지막 입력 상태
        last_input = x[:, -1, :]
        
        # 출력 레이어 (잔차 연결 포함)
        combined = torch.cat([normed, last_input], dim=1)
        output = self.output_layers(combined)
        output = output * self.output_scale

        return output


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
    Xt_norm = (Xt - mu_X) / sigma_X
    Xth1_norm = (Xth1 - mu_X) / sigma_X
    Xth2_norm = (Xth2 - mu_X) / sigma_X
    
    # LSTM 입력을 위한 시퀀스 구성
    sequence = torch.stack([Xth2_norm, Xth1_norm, Xt_norm], dim=1)
    
    # LSTM forward pass
    output = model(sequence)
    
    # 역정규화
    output = output * sigma_X + mu_X
    
    return output


def forward_pass_10(Xt, Xth1, Xth2, Xth3, Xth4, Xth5, Xth6, Xth7, Xth8, Xth9, Xth10, model, sigma_X, mu_X):
    """
    LSTM을 사용한 forward pass 함수 (10개의 과거 시점 사용)
    시계열 데이터를 과거→현재 순서로 구성하고
    정규화/역정규화 수행
    """
    # float32로 변환
    Xt = Xt.float()
    Xth1 = Xth1.float()
    Xth2 = Xth2.float()
    Xth3 = Xth3.float()
    Xth4 = Xth4.float()
    Xth5 = Xth5.float()
    Xth6 = Xth6.float()
    Xth7 = Xth7.float()
    Xth8 = Xth8.float()
    Xth9 = Xth9.float()
    Xth10 = Xth10.float()
    
    # 정규화
    Xt_norm = (Xt - mu_X) / sigma_X
    Xth1_norm = (Xth1 - mu_X) / sigma_X
    Xth2_norm = (Xth2 - mu_X) / sigma_X
    Xth3_norm = (Xth3 - mu_X) / sigma_X
    Xth4_norm = (Xth4 - mu_X) / sigma_X
    Xth5_norm = (Xth5 - mu_X) / sigma_X
    Xth6_norm = (Xth6 - mu_X) / sigma_X
    Xth7_norm = (Xth7 - mu_X) / sigma_X
    Xth8_norm = (Xth8 - mu_X) / sigma_X
    Xth9_norm = (Xth9 - mu_X) / sigma_X
    Xth10_norm = (Xth10 - mu_X) / sigma_X
    
    # LSTM 입력을 위한 시퀀스 구성 (과거→현재 순서)
    sequence = torch.stack([
        Xth10_norm, Xth9_norm, Xth8_norm, Xth7_norm, Xth6_norm,
        Xth5_norm, Xth4_norm, Xth3_norm, Xth2_norm, Xth1_norm,
        Xt_norm
    ], dim=1)
    
    # LSTM forward pass
    output = model(sequence)
    
    # 역정규화
    output = output * sigma_X + mu_X
    
    return output


def L96_2t_xdot_10(Xt, Xth1, Xth2, Xth3, Xth4, Xth5, Xth6, Xth7, Xth8, Xth9, Xth10, model, F, sigma_X, mu_X):
    """
    LSTM 모델을 사용한 Lorenz 96 시스템의 시간 미분 계산 (10개 과거 시점 사용)
    """
    # Lorenz 96 시스템의 기본 항
    if isinstance(Xt, np.ndarray):
        Xt = torch.from_numpy(Xt.astype(np.float32))
    
    Xdot = torch.roll(Xt, 1, dims=1) * (torch.roll(Xt, -1, dims=1) - torch.roll(Xt, 2, dims=1)) - Xt + F
    
    # Memory term 추가
    memory_term = forward_pass_10(
        Xt, Xth1, Xth2, Xth3, Xth4, Xth5, Xth6, Xth7, Xth8, Xth9, Xth10,
        model, sigma_X, mu_X
    )
    
    # 텐서 덧셈
    Xdot = Xdot + memory_term
    
    return Xdot  # numpy로 변환하지 않고 tensor 반환


def stepper_10(Xt, model, F, sigma_X, mu_X, dt):
    """
    Xt: (batch_size, 2*n_hist+2, K)
    model: LSTMMemoryNetwork 인스턴스
    F: float
    sigma_X: (K,)
    mu_X: (K,)
    dt: float

    RK4 방법으로 DDE를 적분하여 현재 시점의 state X를 계산
    """
    # 과거 상태 추출
    past_X10 = Xt[:,-22,:]  # t-10 시점
    past_X9 = Xt[:,-20,:]   # t-9 시점
    past_X8 = Xt[:,-18,:]   # t-8 시점
    past_X7 = Xt[:,-16,:]   # t-7 시점
    past_X6 = Xt[:,-14,:]   # t-6 시점
    past_X5 = Xt[:,-12,:]   # t-5 시점
    past_X4 = Xt[:,-10,:]   # t-4 시점
    past_X3 = Xt[:,-8,:]    # t-3 시점
    past_X2 = Xt[:,-6,:]    # t-2 시점
    past_X1 = Xt[:,-4,:]    # t-1 시점
    current_X = Xt[:,-2,:]  # 현재 시점

    # RK4 방법
    Xdot1 = L96_2t_xdot_10(current_X, past_X1, past_X2, past_X3, past_X4, past_X5,
                           past_X6, past_X7, past_X8, past_X9, past_X10,
                           model, F, sigma_X, mu_X)
    
    Xdot2 = L96_2t_xdot_10(current_X + 0.5 * dt * Xdot1,
                           past_X1, past_X2, past_X3, past_X4, past_X5,
                           past_X6, past_X7, past_X8, past_X9, past_X10,
                           model, F, sigma_X, mu_X)
    
    Xdot3 = L96_2t_xdot_10(current_X + 0.5 * dt * Xdot2,
                           past_X1, past_X2, past_X3, past_X4, past_X5,
                           past_X6, past_X7, past_X8, past_X9, past_X10,
                           model, F, sigma_X, mu_X)
    
    Xdot4 = L96_2t_xdot_10(current_X + dt * Xdot3,
                           current_X, past_X1, past_X2, past_X3, past_X4,
                           past_X5, past_X6, past_X7, past_X8, past_X9,
                           model, F, sigma_X, mu_X)

    # 최종 상태 계산
    X_future = current_X + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
    
    return X_future

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



def evaluate_forward_pass_10(Xt, Xth1, Xth2, Xth3, Xth4, Xth5, Xth6, Xth7, Xth8, Xth9, Xth10, model, sigma_X, mu_X):
    """
    LSTM을 사용한 forward pass 평가 함수 (10개의 과거 시점 사용)
    
    Args:
        Xt, Xth1~Xth10 (torch.Tensor/np.ndarray): 현재 및 과거 10개 시점의 상태
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
        Xth3 = torch.from_numpy(Xth3.astype(np.float32))
        Xth4 = torch.from_numpy(Xth4.astype(np.float32))
        Xth5 = torch.from_numpy(Xth5.astype(np.float32))
        Xth6 = torch.from_numpy(Xth6.astype(np.float32))
        Xth7 = torch.from_numpy(Xth7.astype(np.float32))
        Xth8 = torch.from_numpy(Xth8.astype(np.float32))
        Xth9 = torch.from_numpy(Xth9.astype(np.float32))
        Xth10 = torch.from_numpy(Xth10.astype(np.float32))
        
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
        Hh3 = (Xth3 - mu_X) / sigma_X
        Hh4 = (Xth4 - mu_X) / sigma_X
        Hh5 = (Xth5 - mu_X) / sigma_X
        Hh6 = (Xth6 - mu_X) / sigma_X
        Hh7 = (Xth7 - mu_X) / sigma_X
        Hh8 = (Xth8 - mu_X) / sigma_X
        Hh9 = (Xth9 - mu_X) / sigma_X
        Hh10 = (Xth10 - mu_X) / sigma_X
        
        # LSTM 입력을 위한 시퀀스 구성 (과거→현재 순서)
        sequence = torch.stack([
            Hh10, Hh9, Hh8, Hh7, Hh6,
            Hh5, Hh4, Hh3, Hh2, Hh1,
            H
        ], dim=1)
        
        # LSTM forward pass
        output = model(sequence)
        
        # 역정규화
        output = output * sigma_X + mu_X
        
    return output



def evaluate_L96_2t_xdot_10(Xt, Xth1, Xth2, Xth3, Xth4, Xth5, Xth6, Xth7, Xth8, Xth9, Xth10, model, sigma_X, mu_X, F):    
    """
    LSTM 모델을 사용한 Lorenz 96 시스템의 시간 미분 계산 (10개 과거 시점 사용)
    """
    # numpy 입력을 float32 tensor로 변환
    if isinstance(Xt, np.ndarray):
        Xt = torch.from_numpy(Xt.astype(np.float32))
    
    # Lorenz 96 시스템의 기본 항
    Xdot = torch.roll(Xt, 1, dims=1) * (torch.roll(Xt, -1, dims=1) - torch.roll(Xt, 2, dims=1)) - Xt + F
    
    # Memory term 추가
    memory_term = evaluate_forward_pass_10(
        Xt, Xth1, Xth2, Xth3, Xth4, Xth5, Xth6, Xth7, Xth8, Xth9, Xth10,
        model, sigma_X, mu_X
    )
    
    # 텐서 덧셈
    Xdot = Xdot + memory_term
    
    return Xdot

def integrate_L96_2t_with_NN_10(X0, si, nt, model, F, sigma_X, mu_X, t0=0, dt=0.001):
    """
    10개의 과거 시점을 사용하는 LSTM 모델을 이용한 Lorenz 96 시스템 적분
    """
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
            # RK4 update of X
            Xdot1 = evaluate_L96_2t_xdot_10(
                xhist[-2][None,:], xhist[-4][None,:], xhist[-6][None,:], xhist[-8][None,:], xhist[-10][None,:],
                xhist[-12][None,:], xhist[-14][None,:], xhist[-16][None,:], xhist[-18][None,:], xhist[-20][None,:],
                xhist[-22][None,:], model, sigma_X, mu_X, F
            )
            
            Xdot2 = evaluate_L96_2t_xdot_10(
                xhist[-2][None,:] + 0.5 * dt * Xdot1, xhist[-3][None,:], xhist[-5][None,:], xhist[-7][None,:],
                xhist[-9][None,:], xhist[-11][None,:], xhist[-13][None,:], xhist[-15][None,:], xhist[-17][None,:],
                xhist[-19][None,:], xhist[-21][None,:], model, sigma_X, mu_X, F
            )
            
            Xdot3 = evaluate_L96_2t_xdot_10(
                xhist[-2][None,:] + 0.5 * dt * Xdot2, xhist[-3][None,:], xhist[-5][None,:], xhist[-7][None,:],
                xhist[-9][None,:], xhist[-11][None,:], xhist[-13][None,:], xhist[-15][None,:], xhist[-17][None,:],
                xhist[-19][None,:], xhist[-21][None,:], model, sigma_X, mu_X, F
            )
            
            Xdot4 = evaluate_L96_2t_xdot_10(
                xhist[-2][None,:] + dt * Xdot3, xhist[-2][None,:], xhist[-4][None,:], xhist[-6][None,:],
                xhist[-8][None,:], xhist[-10][None,:], xhist[-12][None,:], xhist[-14][None,:], xhist[-16][None,:],
                xhist[-18][None,:], xhist[-20][None,:], model, sigma_X, mu_X, F
            )
            
            X = xhist[-2][None,:] + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
            
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
    model = CNNLSTMMemoryNetwork(
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
    eval_model = CNNLSTMMemoryNetwork(
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
        NN_out = evaluate_forward_pass_2(Xt_dt, Xth1_dt, Xth2_dt, eval_model, sigma_X, mu_X)
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
        
        # 모델 출력 계산
        NN_out = evaluate_forward_pass_10(
            Xt_dt, Xth1_dt, Xth2_dt, Xth3_dt, Xth4_dt, 
            Xth5_dt, Xth6_dt, Xth7_dt, Xth8_dt, Xth9_dt, 
            Xth10_dt, eval_model, sigma_X, mu_X
        )
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
        NN_out_ext = evaluate_forward_pass_2(Xt_dt, Xth1_dt, Xth2_dt, eval_model, sigma_X, mu_X)
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
        
        # 모델 출력 계산
        NN_out_ext = evaluate_forward_pass_10(
            Xt_dt, Xth1_dt, Xth2_dt, Xth3_dt, Xth4_dt, 
            Xth5_dt, Xth6_dt, Xth7_dt, Xth8_dt, Xth9_dt, 
            Xth10_dt, eval_model, sigma_X, mu_X
        )
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
    