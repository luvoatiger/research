import os
import sys

# OpenMP 중복 라이브러리 로드 문제 해결
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

class dAMZ(nn.Module):
    """
    d-AMZ (discrete Approximated Mori Zwanzig) 신경망
    논문 Section 3.3의 수식에 따른 구현
    
    수정된 버전:
    - Markov term: Lorenz 96 시스템의 실제 동역학 방정식으로 계산
    - Memory term: 신경망으로 학습
    
    Eq. (13): D = d × (n_M + 1)
    Eq. (14): Z = (z_n^T, z_{n-1}^T, ..., z_{n-nM}^T)^T ∈ R^D
    Eq. (15): N(⋅; Θ) : R^D → R^d (memory term만)
    Eq. (16): z^out = z_n + dt * markov_term + N(Z^in)
    Eq. (17): z_{n+1} = z_n + dt * markov_term + N(z_n, z_{n-1}, ..., z_{n-nM}; Θ), n ≥ n_M
    """

    def __init__(self, d=3, n_M=60, hidden_dim=30, dropout_rate=0.1, F=8.0, dt=0.005):
        """
        dAMZ 신경망 초기화

        Args:
            d (int): 축소된 변수의 차원 (기본값: 3, x1, x2, x3)
            n_M (int): 메모리 항목 수 (memory_length_TM / dt)
            hidden_dim (int): 은닉층 차원
            dropout_rate (float): dropout 비율 (기본값: 0.1)
            F (float): Lorenz 96 시스템의 강제력 파라미터 (기본값: 8.0)
            dt (float): 시간 간격 (기본값: 0.005)
        """
        super(dAMZ, self).__init__()

        self.d = d  # 축소된 변수 차원
        self.n_M = n_M  # 메모리 항목 수
        self.D = d * (n_M + 1)  # Eq. (13): D = d × (n_M + 1)
        self.dropout_rate = dropout_rate
        self.F = F  # Lorenz 96 강제력
        self.dt = dt  # 시간 간격

        # 신경망 N(⋅; Θ) 정의: R^D → R^d (memory term만 학습)
        self.neural_network = nn.Sequential(
            nn.Linear(self.D, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 첫 번째 dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 두 번째 dropout
            nn.Linear(hidden_dim, d)
        )

        # 가중치 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """가중치 초기화 - 더 안정적인 초기화"""
        if isinstance(module, nn.Linear):
            # Xavier 초기화 대신 더 작은 값으로 초기화
            torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def lorenz96_markov_term(self, z_n):
        """
        Lorenz 96 시스템의 Markov term 계산
        d/dt X[k] = (X[k+1] - X[k-2]) * X[k-1] - X[k] + F
        
        Args:
            z_n (torch.Tensor): 현재 상태 [batch_size, d]
            
        Returns:
            torch.Tensor: Markov term [batch_size, d]
        """
        batch_size = z_n.shape[0]
        
        # 순환 인덱싱을 위한 roll 연산
        roll_p1 = torch.roll(z_n, shifts=-1, dims=1)  # X[k+1]
        roll_m2 = torch.roll(z_n, shifts=2, dims=1)   # X[k-2]
        roll_m1 = torch.roll(z_n, shifts=1, dims=1)   # X[k-1]
        
        # Lorenz 96 방정식: dX/dt = (X[k+1] - X[k-2]) * X[k-1] - X[k] + F
        markov_term = (roll_p1 - roll_m2) * roll_m1 - z_n + self.F
        
        return markov_term

    def system_dynamics(self, z_n, Z_in, enable_dropout=True):
        """
        시스템의 전체 동역학 계산: dz/dt = markov_term + memory_term
        
        Args:
            z_n (torch.Tensor): 현재 상태 [batch_size, d]
            Z_in (torch.Tensor): 전체 입력 텐서 [batch_size, D]
            enable_dropout (bool): Dropout 활성화 여부
            
        Returns:
            torch.Tensor: 전체 미분 [batch_size, d]
        """
        markov_term = self.lorenz96_markov_term(z_n)
        
        if enable_dropout:
            self.neural_network.train()
        else:
            self.neural_network.eval()
            
        memory_term = self.neural_network(Z_in)
        
        return markov_term + memory_term

    def rk4_step(self, z_n, Z_in, enable_dropout=True):
        """
        RK4 (Runge-Kutta 4th order) 수치적분을 사용한 한 스텝 진행
        
        Args:
            z_n (torch.Tensor): 현재 상태 [batch_size, d]
            Z_in (torch.Tensor): 전체 입력 텐서 [batch_size, D]
            enable_dropout (bool): Dropout 활성화 여부
            
        Returns:
            torch.Tensor: 다음 상태 [batch_size, d]
        """
        dt = self.dt
        
        # RK4 계수 계산
        k1 = self.system_dynamics(z_n, Z_in, enable_dropout)
        k2 = self.system_dynamics(z_n + 0.5 * dt * k1, Z_in, enable_dropout)
        k3 = self.system_dynamics(z_n + 0.5 * dt * k2, Z_in, enable_dropout)
        k4 = self.system_dynamics(z_n + dt * k3, Z_in, enable_dropout)
        
        # RK4 공식 적용
        z_next = z_n + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        
        return z_next

    def forward(self, Z_in, enable_dropout=True):
        """
        Forward pass
        
        Args:
            Z_in (torch.Tensor): 입력 텐서 [batch_size, D]
            enable_dropout (bool): Dropout 활성화 여부
            
        Returns:
            torch.Tensor: 출력 텐서 [batch_size, d]
        """
        batch_size = Z_in.shape[0]
        I_hat = torch.zeros(self.d, self.D)
        I_hat[:self.d, :self.d] = torch.eye(self.d)
        I_hat_batch = I_hat.unsqueeze(0).expand(batch_size, -1, -1)
        Z_in_matrix = Z_in.unsqueeze(2)
        z_n = torch.bmm(I_hat_batch, Z_in_matrix).squeeze(2)

        # Euler 적분 사용
        markov_term = self.lorenz96_markov_term(z_n)

        if enable_dropout:
            self.neural_network.train()
        else:
            self.neural_network.eval()
            
        memory_term = self.neural_network(Z_in)

        z_out = z_n + self.dt * (markov_term + memory_term)

        return z_out

    def predict_with_uncertainty(self, Z_in, num_samples=100):
        """
        Monte Carlo Dropout을 사용한 불확실성 추정
        
        Args:
            Z_in (torch.Tensor): 입력 텐서 [batch_size, D]
            num_samples (int): Monte Carlo 샘플 수
            
        Returns:
            tuple: (mean_prediction, std_prediction)
                - mean_prediction: 평균 예측 [batch_size, d]
                - std_prediction: 예측 표준편차 [batch_size, d]
        """
        self.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                pred = self.forward(Z_in, enable_dropout=True)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        mean_prediction = torch.mean(predictions, dim=0)
        std_prediction = torch.std(predictions, dim=0)
        
        return mean_prediction, std_prediction

    def get_memory_structure_info(self):
        """메모리 구조 정보 반환"""
        return {
            'd': self.d,
            'n_M': self.n_M,
            'D': self.D,
            'input_dim': self.D,
            'output_dim': self.d,
            'dropout_rate': self.dropout_rate,
            'F': self.F,
            'dt': self.dt,
            'integration_method': 'RK4 (default) / Euler (optional)'
        }


def create_training_dataset(all_trajectories, memory_range_NM, selection_mode='random', J0=5):
    if selection_mode == 'random':
        return create_random_training_dataset(all_trajectories, memory_range_NM, J0)
    elif selection_mode == 'deterministic':
        return create_deterministic_training_dataset(all_trajectories, memory_range_NM)
    else:
        raise ValueError(f"Invalid selection mode: {selection_mode}")


def create_random_training_dataset(all_trajectories, memory_range_NM, J0=5):
    Z_list, z_list = [], []
    input_len = memory_range_NM + 1
    total_len = memory_range_NM + 2

    for traj in all_trajectories:
        max_start = len(traj) - total_len
        if max_start < 1:
            continue
        J_i = min(J0, max_start)
        start_indices = np.random.choice(max_start + 1, J_i, replace=False)
        for start in start_indices:
            Z_j_i = traj[start:start + input_len].reshape(-1)
            z_j_i = traj[start + input_len]
            Z_list.append(Z_j_i)
            z_list.append(z_j_i)

    Z = np.array(Z_list)
    z = np.array(z_list)
    print(f"[Random] Z.shape = {Z.shape}, z.shape = {z.shape}")
    return Z, z


def create_deterministic_training_dataset(all_trajectories, memory_range_NM):
    Z_list, z_list = [], []
    input_len = memory_range_NM + 1
    total_len = memory_range_NM + 2

    for traj in all_trajectories:
        max_start = len(traj) - total_len
        if max_start < 1:
            continue
        for start in range(max_start + 1):
            Z_j_i = traj[start:start + input_len].reshape(-1)
            z_j_i = traj[start + input_len]
            Z_list.append(Z_j_i)
            z_list.append(z_j_i)

    Z = np.array(Z_list)
    z = np.array(z_list)
    print(f"[Deterministic] Z.shape = {Z.shape}, z.shape = {z.shape}")
    return Z, z


def normalize_data(Z, z):
    """
    데이터 정규화 함수
    
    Args:
        Z: 입력 데이터 [N, D]
        z: 타겟 데이터 [N, d]
    
    Returns:
        Z_norm: 정규화된 입력 데이터
        z_norm: 정규화된 타겟 데이터
        Z_mean, Z_std: 입력 데이터의 평균과 표준편차
        z_mean, z_std: 타겟 데이터의 평균과 표준편차
    """
    # 입력 데이터 정규화
    Z_mean = np.mean(Z, axis=0)
    Z_std = np.std(Z, axis=0)
    Z_std = np.where(Z_std == 0, 1.0, Z_std)  # 표준편차가 0인 경우 1로 설정
    Z_norm = (Z - Z_mean) / Z_std
    
    # 타겟 데이터 정규화
    z_mean = np.mean(z, axis=0)
    z_std = np.std(z, axis=0)
    z_std = np.where(z_std == 0, 1.0, z_std)  # 표준편차가 0인 경우 1로 설정
    z_norm = (z - z_mean) / z_std
    
    print(f"입력 데이터 통계: mean={Z_mean.mean():.4f}, std={Z_std.mean():.4f}")
    print(f"타겟 데이터 통계: mean={z_mean.mean():.4f}, std={z_std.mean():.4f}")
    
    return Z_norm, z_norm, Z_mean, Z_std, z_mean, z_std


def train_model(model, Z, z, epochs=2000, lr=1e-3, batch_size=32, normalize=True, clip_grad_norm=1.0):
    """
    dAMZ 모델 학습 함수 (배치 처리 포함)

    Args:
        model: dAMZ 모델
        Z: 입력 데이터 [N, D]
        z: 타겟 데이터 [N, d]
        epochs: 학습 에포크 수
        lr: 학습률
        batch_size: 배치 크기
        normalize: 데이터 정규화 여부
        clip_grad_norm: 그래디언트 클리핑 값
    """
    # 데이터 정규화
    if normalize:
        Z_norm, z_norm, Z_mean, Z_std, z_mean, z_std = normalize_data(Z, z)
        Z_tensor = torch.FloatTensor(Z_norm)
        z_tensor = torch.FloatTensor(z_norm)
    else:
        Z_tensor = torch.FloatTensor(Z)
        z_tensor = torch.FloatTensor(z)

    # 데이터셋 생성
    dataset = torch.utils.data.TensorDataset(Z_tensor, z_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 손실 함수와 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # L2 정규화 추가
    
    # 학습률 스케줄러 추가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)

    # 학습
    model.train()
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 100

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_Z, batch_z in dataloader:
            optimizer.zero_grad()

            # 순전파
            outputs = model(batch_Z)
            loss = criterion(outputs, batch_z)

            # NaN 체크
            if torch.isnan(loss):
                print(f"경고: Epoch {epoch+1}에서 NaN 손실이 발생했습니다!")
                print(f"입력 데이터 범위: [{batch_Z.min():.4f}, {batch_Z.max():.4f}]")
                print(f"타겟 데이터 범위: [{batch_z.min():.4f}, {batch_z.max():.4f}]")
                print(f"출력 데이터 범위: [{outputs.min():.4f}, {outputs.max():.4f}]")
                return losses

            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            # 그래디언트 NaN 체크
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"경고: {name}에서 NaN 그래디언트가 발생했습니다!")
                    return losses
            
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # 에포크 평균 손실 계산
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(avg_loss)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')

    print(f'Training completed. Final loss: {losses[-1]:.6f}')
    return losses


def lorenz96_system_equations(t, state, F, h, b, c, K, J):
    """
    Lorenz 96 two-time-scale 시스템의 미분 방정식
    
    Args:
        t: 시간
        state: 상태 벡터 [X, Y] (X는 K차원, Y는 K*J차원)
        F, h, b, c: 시스템 파라미터
        K: X 변수의 수
        J: 각 X에 대응하는 Y 변수의 수
    
    Returns:
        derivatives: 미분값 [dX/dt, dY/dt]
    """
    X = state[:K]
    Y = state[K:]
    
    # X의 미분 계산
    dX = np.zeros(K)
    for k in range(K):
        dX[k] = (X[(k+1) % K] - X[(k-2) % K]) * X[(k-1) % K] - X[k] + F - (h * c / b) * np.sum(Y[k*J:(k+1)*J])
    
    # Y의 미분 계산
    dY = np.zeros(K * J)
    for k in range(K):
        for j in range(J):
            idx = k * J + j
            dY[idx] = -b * c * Y[(idx + 1) % (K * J)] * (Y[(idx + 2) % (K * J)] - Y[(idx - 1) % (K * J)]) - c * Y[idx] + (h * c / b) * X[k]
    
    return np.concatenate([dX, dY])


def simulate_and_plot_lorenz96_x1_prediction(model, metadata, memory_length_TM, X_init, Y_init, t_end=10, t_start_plot=2, delta=0.005):
    """
    Lorenz 96 시스템의 첫 번째 변수(X1)에 대한 모델 예측과 실제 시뮬레이션을 비교하여 시각화하는 함수

    Args:
        model: 학습된 dAMZ 모델
        metadata: Lorenz 96 시스템의 메타데이터
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
    """
    # t_end가 t_start_plot보다 큰지 확인하고 예외 처리
    if t_end <= t_start_plot:
        print(f"경고: t_end({t_end})가 t_start_plot({t_start_plot})보다 작거나 같습니다.")
        print(f"t_end를 {t_start_plot + 1.0}로 조정합니다.")
        t_end = t_start_plot + 1.0
    
    # 시스템 파라미터 추출
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    
    # 초기 조건 설정 (Lorenz 96 시스템에 맞는 초기 조건)
    k = np.arange(K)
    j = np.arange(J * K)
    
    # 초기 조건 함수 (multiscale_lorenz.py에서 가져옴)
    def s(k, K):
        return 2 * np.pi * k / K
    
    if X_init is None or Y_init is None:
        X_init = s(k, K) * (s(k, K) - 1) * (s(k, K) + 1)
        Y_init = 0 * s(j, J * K) * (s(j, J * K) - 1) * (s(j, J * K) + 1)
    initial_condition = np.concatenate([X_init, Y_init])

    print(f"초기 조건 X: {X_init}")
    print(f"초기 조건 Y: {Y_init[:J]} (첫 번째 그룹만 표시)")

    # 실제 시뮬레이션
    t_eval = np.arange(0, t_end, delta)  # t_end는 포함하지 않음
    t_span = (0, t_end)

    try:
        # Lorenz 96 시스템 적분
        sol = solve_ivp(
            fun=lambda t, y: lorenz96_system_equations(t, y, F, h, b, c, K, J),
            t_span=t_span,
            y0=initial_condition,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )

        t_vals = sol.t
        X_true = sol.y[:K, :].T  # X 변수들만 추출
        x1_true = X_true[:, 0]  # 첫 번째 변수

        print(f"시뮬레이션 완료: {len(t_vals)} 시간 스텝")
        print(f"X1 범위: [{x1_true.min():.3f}, {x1_true.max():.3f}]")

        # 시스템 발산 감지
        if np.any(np.abs(x1_true) > 1000) or len(t_vals) < t_end / delta * 0.5:
            print("경고: 시스템이 발산했습니다. 이 초기 조건은 건너뜁니다.")
            return

        # 모델 예측
        model.eval()
        n_M = int(memory_length_TM / dt)  # 메모리 길이 (일관성을 위해 동일하게 설정)

        # 메모리 항목들을 포함한 입력 데이터 준비
        x1_pred = []
        with torch.no_grad():
            for i in range(n_M + 1, len(X_true)):
                # Z = (z_n^T, z_{n-1}^T, ..., z_{n-nM}^T)^T
                Z_input = X_true[i-n_M-1:i].reshape(-1)  # [D]
                Z_tensor = torch.FloatTensor(Z_input).unsqueeze(0)  # [1, D]

                # 예측
                z_pred = model(Z_tensor, enable_dropout=False)
                x1_pred.append(z_pred[0, 0].item())  # X1 예측값

        print(f"예측 완료: {len(x1_pred)} 개의 예측값")

        # 시각화
        t_pred = t_vals[n_M + 1:]
        mask = (t_pred >= t_start_plot)
        t_plot = t_pred[mask]
        x1_true_plot = x1_true[n_M + 1:][mask]
        x1_pred_plot = np.array(x1_pred)[mask]

        print(f"필터링 후 데이터: {len(t_plot)} 개의 시간 포인트")
        print(f"t_plot 범위: [{t_plot.min():.3f}, {t_plot.max():.3f}]")

        if len(t_plot) == 0:
            print("경고: 필터링 후 데이터가 없습니다!")
            print(f"t_start_plot={t_start_plot}, t_pred 범위=[{t_pred.min():.3f}, {t_pred.max():.3f}]")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(t_plot, x1_true_plot, 'b-', label='True $X_1(t)$', linewidth=2)
        plt.plot(t_plot, x1_pred_plot, 'r--', label='Pred $X_1(t)$', linewidth=2)
        plt.xlabel('time t')
        plt.ylabel('$X_1(t)$')
        plt.title(f'Lorenz 96 dAMZ prediction vs real simulation [{t_start_plot}, {t_end}]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # MSE 및 RMSE 계산
        mse = np.mean((x1_true_plot - x1_pred_plot) ** 2)
        rmse = np.sqrt(mse)
        print(f'예측 MSE: {mse:.6f}')
        print(f'예측 RMSE: {rmse:.6f}')
        
        # 데이터 범위 대비 상대적 오차
        data_range = x1_true_plot.max() - x1_true_plot.min()
        relative_error = rmse / data_range * 100
        print(f'상대적 오차: {relative_error:.2f}%')

    except Exception as e:
        print(f"시뮬레이션 중 오류 발생: {e}")
        print(f"초기 조건에서 시스템이 불안정할 수 있습니다.")


def simulate_and_plot_lorenz96_all_variables_prediction(model, metadata, memory_length_TM, X_init, Y_init, t_end=10, t_start_plot=2, delta=0.005):
    """
    Lorenz 96 시스템의 모든 X 변수에 대한 모델 예측과 실제 시뮬레이션을 비교하여 시각화하는 함수

    Args:
        model: 학습된 dAMZ 모델
        metadata: Lorenz 96 시스템의 메타데이터
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
    """
    # t_end가 t_start_plot보다 큰지 확인하고 예외 처리
    if t_end <= t_start_plot:
        print(f"경고: t_end({t_end})가 t_start_plot({t_start_plot})보다 작거나 같습니다.")
        print(f"t_end를 {t_start_plot + 1.0}로 조정합니다.")
        t_end = t_start_plot + 1.0
    
    # 시스템 파라미터 추출
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    
    # 초기 조건 설정
    k = np.arange(K)
    j = np.arange(J * K)
    # 초기 조건 함수 (multiscale_lorenz.py에서 가져옴)
    def s(k, K):
        return 2 * np.pi * k / K

    if X_init is None or Y_init is None:
        X_init = s(k, K) * (s(k, K) - 1) * (s(k, K) + 1)
        Y_init = 0 * s(j, J * K) * (s(j, J * K) - 1) * (s(j, J * K) + 1)
        
    initial_condition = np.concatenate([X_init, Y_init])

    # 실제 시뮬레이션
    t_eval = np.arange(0, t_end, delta)  # t_end는 포함하지 않음
    t_span = (0, t_end)

    try:
        sol = solve_ivp(
            fun=lambda t, y: lorenz96_system_equations(t, y, F, h, b, c, K, J),
            t_span=t_span,
            y0=initial_condition,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )

        t_vals = sol.t
        X_true = sol.y[:K, :].T

        # 모델 예측
        model.eval()
        n_M = int(memory_length_TM / dt)  # 메모리 길이 (일관성을 위해 동일하게 설정)

        X_pred = []
        with torch.no_grad():
            for i in range(n_M + 1, len(X_true)):
                Z_input = X_true[i-n_M-1:i].reshape(-1)
                Z_tensor = torch.FloatTensor(Z_input).unsqueeze(0)
                z_pred = model(Z_tensor)
                X_pred.append(z_pred[0].numpy())

        X_pred = np.array(X_pred)
        t_pred = t_vals[n_M + 1:]
        mask = (t_pred >= t_start_plot)
        t_plot = t_pred[mask]
        X_true_plot = X_true[n_M + 1:][mask]
        X_pred_plot = X_pred[mask]

        # 모든 변수 시각화
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i in range(K):
            ax = axes[i]
            ax.plot(t_plot, X_true_plot[:, i], 'b-', label='True', linewidth=2)
            ax.plot(t_plot, X_pred_plot[:, i], 'r--', label='Pred', linewidth=2)
            ax.set_xlabel('time t')
            ax.set_ylabel(f'$X_{i+1}(t)$')
            ax.set_title(f'Variable {i+1}')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

        # 전체 MSE 및 RMSE 계산
        mse_total = np.mean((X_true_plot - X_pred_plot) ** 2)
        rmse_total = np.sqrt(mse_total)
        print(f'전체 예측 MSE: {mse_total:.6f}')
        print(f'전체 예측 RMSE: {rmse_total:.6f}')
        
        # 각 변수별 MSE 계산
        for i in range(K):
            mse_var = np.mean((X_true_plot[:, i] - X_pred_plot[:, i]) ** 2)
            rmse_var = np.sqrt(mse_var)
            print(f'변수 {i+1} MSE: {mse_var:.6f}, RMSE: {rmse_var:.6f}')

    except Exception as e:
        print(f"시뮬레이션 중 오류 발생: {e}")


def simulate_and_plot_lorenz96_x1_prediction_with_uncertainty(model, metadata, memory_length_TM, X_init, Y_init, t_end=10, t_start_plot=2, delta=0.005, num_mc_samples=100):
    """
    Lorenz 96 시스템의 첫 번째 변수(X1)에 대한 불확실성을 포함한 모델 예측과 실제 시뮬레이션을 비교하여 시각화하는 함수

    Args:
        model: 학습된 dAMZ 모델 (Monte Carlo Dropout 포함)
        metadata: Lorenz 96 시스템의 메타데이터
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
        num_mc_samples: Monte Carlo 샘플 수
    """
    # t_end가 t_start_plot보다 큰지 확인하고 예외 처리
    if t_end <= t_start_plot:
        print(f"경고: t_end({t_end})가 t_start_plot({t_start_plot})보다 작거나 같습니다.")
        print(f"t_end를 {t_start_plot + 1.0}로 조정합니다.")
        t_end = t_start_plot + 1.0
    
    # 시스템 파라미터 추출
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    
    # 초기 조건 설정 (Lorenz 96 시스템에 맞는 초기 조건)
    k = np.arange(K)
    j = np.arange(J * K)
    def s(k, K):
        return 2 * np.pi * k / K

    if X_init is None or Y_init is None:
        X_init = s(k, K) * (s(k, K) - 1) * (s(k, K) + 1)
        Y_init = 0 * s(j, J * K) * (s(j, J * K) - 1) * (s(j, J * K) + 1)
    
    initial_condition = np.concatenate([X_init, Y_init])
    print(f"초기 조건 X: {X_init}")
    print(f"초기 조건 Y: {Y_init[:J]} (첫 번째 그룹만 표시)")

    # 실제 시뮬레이션
    t_eval = np.arange(0, t_end, delta)  # t_end는 포함하지 않음
    t_span = (0, t_end)

    try:
        # Lorenz 96 시스템 적분
        sol = solve_ivp(
            fun=lambda t, y: lorenz96_system_equations(t, y, F, h, b, c, K, J),
            t_span=t_span,
            y0=initial_condition,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )

        t_vals = sol.t
        X_true = sol.y[:K, :].T  # X 변수들만 추출
        x1_true = X_true[:, 0]  # 첫 번째 변수

        print(f"시뮬레이션 완료: {len(t_vals)} 시간 스텝")
        print(f"X1 범위: [{x1_true.min():.3f}, {x1_true.max():.3f}]")

        # 시스템 발산 감지
        if np.any(np.abs(x1_true) > 1000) or len(t_vals) < t_end / delta * 0.5:
            print("경고: 시스템이 발산했습니다. 이 초기 조건은 건너뜁니다.")
            return

        # 모델 예측 (불확실성 포함)
        n_M = int(memory_length_TM / dt)  # 메모리 길이 (일관성을 위해 동일하게 설정)

        # 메모리 항목들을 포함한 입력 데이터 준비
        x1_pred_mean = []
        x1_pred_std = []
        
        with torch.no_grad():
            for i in range(n_M + 1, len(X_true)):
                # Z = (z_n^T, z_{n-1}^T, ..., z_{n-nM}^T)^T
                Z_input = X_true[i-n_M-1:i].reshape(-1)  # [D]
                Z_tensor = torch.FloatTensor(Z_input).unsqueeze(0)  # [1, D]

                # Monte Carlo Dropout을 사용한 불확실성 추정
                mean_pred, std_pred = model.predict_with_uncertainty(Z_tensor, num_samples=num_mc_samples)
                x1_pred_mean.append(mean_pred[0, 0].item())  # X1 평균 예측값
                x1_pred_std.append(std_pred[0, 0].item())    # X1 표준편차

        print(f"예측 완료: {len(x1_pred_mean)} 개의 예측값 (Monte Carlo 샘플 수: {num_mc_samples})")

        # 시각화 - 예측값과 실제값이 같은 시점에서 시작하도록 수정
        t_pred = t_vals[n_M + 1:]
        mask = (t_pred >= t_start_plot)
        t_plot = t_pred[mask]
        x1_true_plot = x1_true[n_M + 1:][mask]  # 예측 가능한 시점부터의 실제값
        x1_pred_mean_plot = np.array(x1_pred_mean)[mask]
        x1_pred_std_plot = np.array(x1_pred_std)[mask]

        print(f"필터링 후 데이터: {len(t_plot)} 개의 시간 포인트")
        print(f"t_plot 범위: [{t_plot.min():.3f}, {t_plot.max():.3f}]")

        if len(t_plot) == 0:
            print("경고: 필터링 후 데이터가 없습니다!")
            print(f"t_start_plot={t_start_plot}, t_pred 범위=[{t_pred.min():.3f}, {t_pred.max():.3f}]")
            return

        # 불확실성을 포함한 시각화
        plt.figure(figsize=(12, 8))
        
        # 실제 값과 평균 예측
        plt.plot(t_plot, x1_true_plot, 'b-', label='True $X_1(t)$', linewidth=2)
        plt.plot(t_plot, x1_pred_mean_plot, 'r--', label='Pred $X_1(t)$ (mean)', linewidth=2)
        
        # 불확실성 구간 (95% 신뢰구간)
        plt.fill_between(t_plot, 
                        x1_pred_mean_plot - 2*x1_pred_std_plot, 
                        x1_pred_mean_plot + 2*x1_pred_std_plot, 
                        alpha=0.1, color='red', label='95% Confidence Interval')
        
        # 1 표준편차 구간
        plt.fill_between(t_plot, 
                        x1_pred_mean_plot - x1_pred_std_plot, 
                        x1_pred_mean_plot + x1_pred_std_plot, 
                        alpha=0.1, color='red', label='±1σ Interval')
        
        plt.xlabel('time t')
        plt.ylabel('$X_1(t)$')
        plt.title(f'Lorenz 96 dAMZ prediction with uncertainty [{t_start_plot}, {t_end}]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # MSE 및 RMSE 계산 (평균 예측 기준)
        mse = np.mean((x1_true_plot - x1_pred_mean_plot) ** 2)
        rmse = np.sqrt(mse)
        print(f'예측 MSE: {mse:.6f}')
        print(f'예측 RMSE: {rmse:.6f}')
        
        # 데이터 범위 대비 상대적 오차
        data_range = x1_true_plot.max() - x1_true_plot.min()
        relative_error = rmse / data_range * 100
        print(f'상대적 오차: {relative_error:.2f}%')
        
        # 불확실성 통계
        mean_uncertainty = np.mean(x1_pred_std_plot)
        max_uncertainty = np.max(x1_pred_std_plot)
        print(f'평균 불확실성 (표준편차): {mean_uncertainty:.6f}')
        print(f'최대 불확실성 (표준편차): {max_uncertainty:.6f}')

    except Exception as e:
        print(f"시뮬레이션 중 오류 발생: {e}")
        print(f"초기 조건에서 시스템이 불안정할 수 있습니다.")


def simulate_and_plot_lorenz96_all_variables_prediction_with_uncertainty(model, metadata, memory_length_TM, X_init, Y_init, t_end=10, t_start_plot=2, delta=0.005, num_mc_samples=100):
    """
    Lorenz 96 시스템의 모든 X 변수에 대한 불확실성을 포함한 모델 예측과 실제 시뮬레이션을 비교하여 시각화하는 함수

    Args:
        model: 학습된 dAMZ 모델 (Monte Carlo Dropout 포함)
        metadata: Lorenz 96 시스템의 메타데이터
        X_init: X 변수들의 초기 조건
        Y_init: Y 변수들의 초기 조건
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
        num_mc_samples: Monte Carlo 샘플 수
    """
    # t_end가 t_start_plot보다 큰지 확인하고 예외 처리
    if t_end <= t_start_plot:
        print(f"경고: t_end({t_end})가 t_start_plot({t_start_plot})보다 작거나 같습니다.")
        print(f"t_end를 {t_start_plot + 1.0}로 조정합니다.")
        t_end = t_start_plot + 1.0
    
    # 시스템 파라미터 추출
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    
    # 초기 조건 설정
    k = np.arange(K)
    j = np.arange(J * K)
    # 초기 조건 함수 (multiscale_lorenz.py에서 가져옴)
    def s(k, K):
        return 2 * np.pi * k / K

    if X_init is None or Y_init is None:
        X_init = s(k, K) * (s(k, K) - 1) * (s(k, K) + 1)
        Y_init = 0 * s(j, J * K) * (s(j, J * K) - 1) * (s(j, J * K) + 1)

    initial_condition = np.concatenate([X_init, Y_init])
    print(f"초기 조건 X: {X_init}")
    print(f"초기 조건 Y: {Y_init[:J]} (첫 번째 그룹만 표시)")

    # 실제 시뮬레이션
    t_eval = np.arange(0, t_end, delta)  # t_end는 포함하지 않음
    t_span = (0, t_end)

    try:
        sol = solve_ivp(
            fun=lambda t, y: lorenz96_system_equations(t, y, F, h, b, c, K, J),
            t_span=t_span,
            y0=initial_condition,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )

        t_vals = sol.t
        X_true = sol.y[:K, :].T

        # 시스템 발산 감지
        if np.any(np.abs(X_true) > 1000) or len(t_vals) < t_end / delta * 0.5:
            print("경고: 시스템이 발산했습니다. 이 초기 조건은 건너뜁니다.")
            return

        # 모델 예측 (불확실성 포함)
        n_M = int(memory_length_TM / dt)  # 메모리 길이 (일관성을 위해 동일하게 설정)

        X_pred_mean = []
        X_pred_std = []
        
        with torch.no_grad():
            for i in range(n_M + 1, len(X_true)):
                Z_input = X_true[i-n_M-1:i].reshape(-1)
                Z_tensor = torch.FloatTensor(Z_input).unsqueeze(0)
                
                # Monte Carlo Dropout을 사용한 불확실성 추정
                mean_pred, std_pred = model.predict_with_uncertainty(Z_tensor, num_samples=num_mc_samples)
                X_pred_mean.append(mean_pred[0].numpy())
                X_pred_std.append(std_pred[0].numpy())

            X_pred_mean = np.array(X_pred_mean)
            X_pred_std = np.array(X_pred_std)
            t_pred = t_vals[n_M + 1:]
            mask = (t_pred >= t_start_plot)
            t_plot = t_pred[mask]
            X_true_plot = X_true[n_M + 1:][mask]
            X_pred_mean_plot = X_pred_mean[mask]
            X_pred_std_plot = X_pred_std[mask]

        print(f"예측 완료: {len(X_pred_mean)} 개의 예측값 (Monte Carlo 샘플 수: {num_mc_samples})")

        # 모든 변수 시각화 (불확실성 포함)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i in range(K):
            ax = axes[i]
            
            # 실제 값과 평균 예측
            ax.plot(t_plot, X_true_plot[:, i], 'b-', label='True', linewidth=2)
            ax.plot(t_plot, X_pred_mean_plot[:, i], 'r--', label='Pred (mean)', linewidth=2)
            
            # 불확실성 구간 (95% 신뢰구간)
            ax.fill_between(t_plot, 
                           X_pred_mean_plot[:, i] - 2*X_pred_std_plot[:, i], 
                           X_pred_mean_plot[:, i] + 2*X_pred_std_plot[:, i], 
                           alpha=0.1, color='red', label='95% Confidence')
            
            # 1 표준편차 구간
            ax.fill_between(t_plot, 
                           X_pred_mean_plot[:, i] - X_pred_std_plot[:, i], 
                           X_pred_mean_plot[:, i] + X_pred_std_plot[:, i], 
                           alpha=0.2, color='red', label='±1σ')
            
            ax.set_xlabel('time t')
            ax.set_ylabel(f'$X_{i+1}(t)$')
            ax.set_title(f'Variable {i+1} with Uncertainty')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

        # 전체 MSE 및 RMSE 계산 (평균 예측 기준)
        mse_total = np.mean((X_true_plot - X_pred_mean_plot) ** 2)
        rmse_total = np.sqrt(mse_total)
        print(f'전체 예측 MSE: {mse_total:.6f}')
        print(f'전체 예측 RMSE: {rmse_total:.6f}')
        
        # 각 변수별 MSE 계산
        for i in range(K):
            mse_var = np.mean((X_true_plot[:, i] - X_pred_mean_plot[:, i]) ** 2)
            rmse_var = np.sqrt(mse_var)
            print(f'변수 {i+1} MSE: {mse_var:.6f}, RMSE: {rmse_var:.6f}')
        
        # 불확실성 통계
        mean_uncertainty = np.mean(X_pred_std_plot)
        max_uncertainty = np.max(X_pred_std_plot)
        print(f'평균 불확실성 (표준편차): {mean_uncertainty:.6f}')
        print(f'최대 불확실성 (표준편차): {max_uncertainty:.6f}')
        
        # 각 변수별 불확실성
        for i in range(K):
            var_mean_uncertainty = np.mean(X_pred_std_plot[:, i])
            var_max_uncertainty = np.max(X_pred_std_plot[:, i])
            print(f'변수 {i+1} 평균 불확실성: {var_mean_uncertainty:.6f}, 최대 불확실성: {var_max_uncertainty:.6f}')

    except Exception as e:
        print(f"시뮬레이션 중 오류 발생: {e}")
        print(f"초기 조건에서 시스템이 불안정할 수 있습니다.")


def evaluate_extrapolation_performance(model, metadata, memory_length_TM, num_trials=5, t_end=10, t_start_plot=2, delta=0.005):
    """
    랜덤 초기 조건에 대해 extrapolation 성능을 평가하는 함수
    논문의 성능 측정 방법: np.linalg.norm(X_ext-Xpred_ext) / np.linalg.norm(X_ext)
    
    Args:
        model: 학습된 dAMZ 모델
        metadata: Lorenz 96 시스템의 메타데이터
        num_trials: 평가할 랜덤 초기 조건의 수
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
    
    Returns:
        performance_scores: 각 시도별 성능 점수 리스트
        mean_performance: 평균 성능 점수
    """
    # t_end가 t_start_plot보다 큰지 확인하고 예외 처리
    if t_end <= t_start_plot:
        print(f"경고: t_end({t_end})가 t_start_plot({t_start_plot})보다 작거나 같습니다.")
        print(f"t_end를 {t_start_plot + 1.0}로 조정합니다.")
        t_end = t_start_plot + 1.0
    
    # 시스템 파라미터 추출
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    
    performance_scores = []
    
    print(f"\n=== Extrapolation 성능 평가 (랜덤 초기 조건 {num_trials}개) ===")
    
    for trial in range(num_trials):
        print(f"\n--- 시도 {trial + 1}/{num_trials} ---")
        
        # 랜덤 초기 조건 생성
        np.random.seed(42 + trial)  # 재현성을 위한 시드 설정
        
        # X 초기 조건: 논문과 유사한 범위로 설정
        X_init = np.random.uniform(-5, 5, K)
        
        # Y 초기 조건: 작은 값으로 설정
        Y_init = np.random.uniform(-0.1, 0.1, K * J)
        
        initial_condition = np.concatenate([X_init, Y_init])
        print(f"초기 조건 X: {X_init}")
        print(f"초기 조건 Y 범위: [{Y_init.min():.3f}, {Y_init.max():.3f}]")

        # 실제 시뮬레이션
        t_eval = np.arange(0, t_end, delta)  # t_end는 포함하지 않음
        t_span = (0, t_end)

        try:
            # Lorenz 96 시스템 적분
            sol = solve_ivp(
                fun=lambda t, y: lorenz96_system_equations(t, y, F, h, b, c, K, J),
                t_span=t_span,
                y0=initial_condition,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-9,
                atol=1e-11
            )

            t_vals = sol.t
            X_true = sol.y[:K, :].T  # X 변수들만 추출

            # 시스템 발산 감지
            if np.any(np.abs(X_true) > 1000) or len(t_vals) < t_end / delta * 0.5:
                print("경고: 시스템이 발산했습니다. 이 시도는 건너뜁니다.")
                continue

            # 모델 예측
            model.eval()
            n_M = int(memory_length_TM / dt)  # 메모리 길이 (일관성을 위해 동일하게 설정)

            X_pred = []
            with torch.no_grad():
                for i in range(n_M + 1, len(X_true)):
                    # Z = (z_n^T, z_{n-1}^T, ..., z_{n-nM}^T)^T
                    Z_input = X_true[i-n_M-1:i].reshape(-1)  # [D]
                    Z_tensor = torch.FloatTensor(Z_input).unsqueeze(0)  # [1, D]

                    # 예측
                    z_pred = model(Z_tensor)
                    X_pred.append(z_pred[0].numpy())

            X_pred = np.array(X_pred)
            t_pred = t_vals[n_M + 1:]
            
            # extrapolation 구간 설정 (t_start_plot 이후)
            mask = (t_pred >= t_start_plot)
            X_ext = X_true[n_M + 1:][mask]  # 실제 extrapolation 데이터
            Xpred_ext = X_pred[mask]  # 예측 extrapolation 데이터

            if len(X_ext) == 0:
                print("경고: extrapolation 데이터가 없습니다!")
                continue

            # 논문의 성능 측정 방법 적용
            # np.linalg.norm(X_ext-Xpred_ext) / np.linalg.norm(X_ext)
            numerator = np.linalg.norm(X_ext - Xpred_ext)
            denominator = np.linalg.norm(X_ext)
            
            if denominator == 0:
                print("경고: 분모가 0입니다!")
                continue
                
            performance_score = numerator / denominator
            performance_scores.append(performance_score)
            
            print(f"Extrapolation 성능 점수: {performance_score:.6f}")
            print(f"  - 분자 (예측 오차 norm): {numerator:.6f}")
            print(f"  - 분모 (실제 데이터 norm): {denominator:.6f}")
            print(f"  - 데이터 포인트 수: {len(X_ext)}")

        except Exception as e:
            print(f"시뮬레이션 중 오류 발생: {e}")
            continue

    # 평균 성능 계산
    if len(performance_scores) > 0:
        mean_performance = np.mean(performance_scores)
        std_performance = np.std(performance_scores)
        
        print(f"\n=== 최종 결과 ===")
        print(f"성공한 시도 수: {len(performance_scores)}/{num_trials}")
        print(f"개별 성능 점수: {[f'{score:.6f}' for score in performance_scores]}")
        print(f"평균 성능 점수: {mean_performance:.6f}")
        print(f"성능 점수 표준편차: {std_performance:.6f}")
        
        return performance_scores, mean_performance
    else:
        print("성공한 시도가 없습니다!")
        return [], None


def evaluate_extrapolation_performance_with_uncertainty(model, metadata, memory_length_TM, num_trials=5, t_end=10, t_start_plot=2, delta=0.005, num_mc_samples=100):
    """
    랜덤 초기 조건에 대해 불확실성을 포함한 extrapolation 성능을 평가하는 함수
    논문의 성능 측정 방법: np.linalg.norm(X_ext-Xpred_ext) / np.linalg.norm(X_ext)
    
    Args:
        model: 학습된 dAMZ 모델 (Monte Carlo Dropout 포함)
        metadata: Lorenz 96 시스템의 메타데이터
        num_trials: 평가할 랜덤 초기 조건의 수
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
        num_mc_samples: Monte Carlo 샘플 수
    
    Returns:
        performance_scores: 각 시도별 성능 점수 리스트
        mean_performance: 평균 성능 점수
        uncertainty_scores: 각 시도별 평균 불확실성 점수 리스트
    """
    # t_end가 t_start_plot보다 큰지 확인하고 예외 처리
    if t_end <= t_start_plot:
        print(f"경고: t_end({t_end})가 t_start_plot({t_start_plot})보다 작거나 같습니다.")
        print(f"t_end를 {t_start_plot + 1.0}로 조정합니다.")
        t_end = t_start_plot + 1.0
    
    # 시스템 파라미터 추출
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    
    performance_scores = []
    uncertainty_scores = []
    
    print(f"\n=== Extrapolation 성능 평가 (불확실성 포함, 랜덤 초기 조건 {num_trials}개) ===")
    
    for trial in range(num_trials):
        print(f"\n--- 시도 {trial + 1}/{num_trials} ---")
        
        # 랜덤 초기 조건 생성
        np.random.seed(42 + trial)  # 재현성을 위한 시드 설정
        
        # X 초기 조건: 논문과 유사한 범위로 설정
        X_init = np.random.uniform(-5, 5, K)
        
        # Y 초기 조건: 작은 값으로 설정
        Y_init = np.random.uniform(-0.1, 0.1, K * J)
        
        initial_condition = np.concatenate([X_init, Y_init])
        print(f"초기 조건 X: {X_init}")
        print(f"초기 조건 Y 범위: [{Y_init.min():.3f}, {Y_init.max():.3f}]")

        # 실제 시뮬레이션
        t_eval = np.arange(0, t_end, delta)  # t_end는 포함하지 않음
        t_span = (0, t_end)

        try:
            # Lorenz 96 시스템 적분
            sol = solve_ivp(
                fun=lambda t, y: lorenz96_system_equations(t, y, F, h, b, c, K, J),
                t_span=t_span,
                y0=initial_condition,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-9,
                atol=1e-11
            )

            t_vals = sol.t
            X_true = sol.y[:K, :].T  # X 변수들만 추출

            # 시스템 발산 감지
            if np.any(np.abs(X_true) > 1000) or len(t_vals) < t_end / delta * 0.5:
                print("경고: 시스템이 발산했습니다. 이 시도는 건너뜁니다.")
                continue

            # 모델 예측 (불확실성 포함)
            n_M = int(memory_length_TM / dt)  # 메모리 길이 (일관성을 위해 동일하게 설정)

            X_pred_mean = []
            X_pred_std = []
            
            with torch.no_grad():
                for i in range(n_M + 1, len(X_true)):
                    # Z = (z_n^T, z_{n-1}^T, ..., z_{n-nM}^T)^T
                    Z_input = X_true[i-n_M-1:i].reshape(-1)  # [D]
                    Z_tensor = torch.FloatTensor(Z_input).unsqueeze(0)  # [1, D]

                    # Monte Carlo Dropout을 사용한 불확실성 추정
                    mean_pred, std_pred = model.predict_with_uncertainty(Z_tensor, num_samples=num_mc_samples)
                    X_pred_mean.append(mean_pred[0].numpy())
                    X_pred_std.append(std_pred[0].numpy())

            X_pred_mean = np.array(X_pred_mean)
            X_pred_std = np.array(X_pred_std)
            t_pred = t_vals[n_M + 1:]
            
            # extrapolation 구간 설정 (t_start_plot 이후)
            mask = (t_pred >= t_start_plot)
            X_ext = X_true[n_M + 1:][mask]  # 실제 extrapolation 데이터
            Xpred_ext = X_pred_mean[mask]  # 예측 extrapolation 데이터 (평균)
            Xpred_std_ext = X_pred_std[mask]  # 예측 불확실성

            if len(X_ext) == 0:
                print("경고: extrapolation 데이터가 없습니다!")
                continue

            # 논문의 성능 측정 방법 적용
            # np.linalg.norm(X_ext-Xpred_ext) / np.linalg.norm(X_ext)
            numerator = np.linalg.norm(X_ext - Xpred_ext)
            denominator = np.linalg.norm(X_ext)
            
            if denominator == 0:
                print("경고: 분모가 0입니다!")
                continue
                
            performance_score = numerator / denominator
            performance_scores.append(performance_score)
            
            # 평균 불확실성 계산
            mean_uncertainty = np.mean(Xpred_std_ext)
            uncertainty_scores.append(mean_uncertainty)
            
            print(f"Extrapolation 성능 점수: {performance_score:.6f}")
            print(f"  - 분자 (예측 오차 norm): {numerator:.6f}")
            print(f"  - 분모 (실제 데이터 norm): {denominator:.6f}")
            print(f"  - 평균 불확실성: {mean_uncertainty:.6f}")
            print(f"  - 데이터 포인트 수: {len(X_ext)}")

        except Exception as e:
            print(f"시뮬레이션 중 오류 발생: {e}")
            continue

    # 평균 성능 계산
    if len(performance_scores) > 0:
        mean_performance = np.mean(performance_scores)
        std_performance = np.std(performance_scores)
        mean_uncertainty = np.mean(uncertainty_scores)
        
        print(f"\n=== 최종 결과 ===")
        print(f"성공한 시도 수: {len(performance_scores)}/{num_trials}")
        print(f"개별 성능 점수: {[f'{score:.6f}' for score in performance_scores]}")
        print(f"평균 성능 점수: {mean_performance:.6f}")
        print(f"성능 점수 표준편차: {std_performance:.6f}")
        print(f"평균 불확실성: {mean_uncertainty:.6f}")
        
        return performance_scores, mean_performance, uncertainty_scores
    else:
        print("성공한 시도가 없습니다!")
        return [], None, []


if __name__ == "__main__":
    # 설정
    hidden_dim = 30
    epochs = 2000
    J0 = 50
    # 기존에 생성한 Lorenz 96 system dataset 이용
    print("\n[+] Loading Lorenz 96 system dataset...")
    results_dir = os.path.join(os.getcwd(), "simulated_data")
    with open(os.path.join(results_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    si = metadata['si']
    spinup_time = metadata['spinup_time']
    forecast_time = metadata['forecast_time']
    num_ic = metadata['num_ic']

    # 메모리 길이를 더 작게 설정하여 예측 시작점을 앞당김
    memory_length_TM = 0.01
    n_M = int(memory_length_TM / dt)
    
    # 데이터 로드 및 전처리
    trajectories = []
    for i in range(1, min(num_ic + 1, 51)):  # 처음 50개만 사용하여 메모리 절약
        try:
            X_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"X_batch_coupled_{i}.npy"))
            # X_data shape: (1, time_steps, 8) -> (time_steps, 8)로 변환
            trajectory = X_data[0]  # 첫 번째 배치만 사용
            trajectory = trajectory[::2,:]
            trajectory = trajectory + 0.003*np.std(trajectory, axis=0)*torch.randn(trajectory.shape, device=torch.device('cpu')).numpy()
            # 데이터 품질 체크
            if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
                print(f"경고: 배치 {i}에서 NaN 또는 Inf 값이 발견되어 건너뜁니다.")
                continue
                
            # 극단적인 값 필터링
            if np.any(np.abs(trajectory) > 1000):
                print(f"경고: 배치 {i}에서 극단적인 값이 발견되어 건너뜁니다.")
                continue
                
            trajectories.append(trajectory)
        except Exception as e:
            print(f"배치 {i} 로드 중 오류: {e}")
            continue

    print(f"로드된 trajectory 수: {len(trajectories)}")            
    print(f"첫 번째 trajectory shape: {trajectories[0].shape}")
    
    Z, z = create_training_dataset(trajectories, n_M, selection_mode='random', J0=J0)
    
    # 데이터 품질 체크
    print(f"Z 데이터 통계: min={Z.min():.4f}, max={Z.max():.4f}, mean={Z.mean():.4f}, std={Z.std():.4f}")
    print(f"z 데이터 통계: min={z.min():.4f}, max={z.max():.4f}, mean={z.mean():.4f}, std={z.std():.4f}")
    
    if np.any(np.isnan(Z)) or np.any(np.isnan(z)):
        print("오류: 학습 데이터에 NaN 값이 포함되어 있습니다!")
        sys.exit(1)

    # Lorenz 96 시스템에 맞는 모델 생성 (Monte Carlo Dropout 포함)
    reduced_order_d = K  # Lorenz 96 시스템의 변수 수
    dropout_rate = 0.1  # dropout 비율
    print(f"모델 생성 시작: reduced_order_d={reduced_order_d}, n_M={n_M}, hidden_dim={hidden_dim}, dropout_rate={dropout_rate}")
    model = dAMZ(d=reduced_order_d, n_M=n_M, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
    print("모델 생성 완료")
    
    # 모델 구조 정보 출력
    model_info = model.get_memory_structure_info()
    print(f"\n[+] dAMZ 모델 구조 (Lorenz 96):")
    print(f"  - 축소된 변수 차원 (d): {model_info['d']}")
    print(f"  - 메모리 항목 수 (n_M): {model_info['n_M']}")
    print(f"  - 입력 차원 (D): {model_info['D']}")
    print(f"  - 출력 차원: {model_info['output_dim']}")
    print(f"  - 실제 입력 데이터 shape: {Z.shape}")
    print(f"  - 실제 출력 데이터 shape: {z.shape}")

    print("\n[+] Training model...")
    # 더 안정적인 학습 파라미터 사용
    losses = train_model(
        model, Z, z, 
        epochs=epochs, 
        lr=5e-4,  # 더 작은 학습률
        batch_size=256,  # 더 작은 배치 크기
        normalize=True,  # 데이터 정규화 활성화
        clip_grad_norm=0.5  # 그래디언트 클리핑
    )
    
    print("\n[+] Lorenz 96 시스템 학습 완료!")
        
    batch_num = np.random.randint(1, 300)
    X_init = np.load(os.path.join(os.getcwd(), "simulated_data", f"ic_X_batch_coupled_{batch_num}.npy"))
    Y_init = np.load(os.path.join(os.getcwd(), "simulated_data", f"ic_Y_batch_coupled_{batch_num}.npy"))
    if X_init.ndim == 2:
        X_init = X_init[0] # shape: [K]
    if Y_init.ndim == 2:
        Y_init = Y_init[0] # shape: [K*J]

    # 예측 시각화
    print("\n[+] Simulating and plotting prediction vs ground truth...")
    
    # 예측 시작 시간 계산
    prediction_start_time = (n_M + 1) * dt  # 실제 예측이 시작되는 시간
    
    # 첫 번째 변수만 시각화
    t_end = 10
    print("\n--- 첫 번째 변수(X1) 예측 ---")
    simulate_and_plot_lorenz96_x1_prediction(model, metadata, memory_length_TM, X_init, Y_init, t_end=t_end, t_start_plot=prediction_start_time, delta=dt)
        
    # 모든 변수 시각화
    print("\n--- 모든 변수 예측 ---")
    simulate_and_plot_lorenz96_all_variables_prediction(model, metadata, memory_length_TM, X_init, Y_init, t_end=t_end, t_start_plot=prediction_start_time, delta=dt)

    # Random IC에 대한 Extrapolation 성능 평가
    print("\n[+] Random IC에 대한 Extrapolation 성능 평가...")
    evaluate_extrapolation_performance(model, metadata, memory_length_TM, num_trials=5, t_end=t_end, t_start_plot=prediction_start_time, delta=dt)

    # 불확실성을 포함한 첫 번째 변수 예측
    print("\n--- 첫 번째 변수(X1) 예측 (불확실성 포함) ---")
    simulate_and_plot_lorenz96_x1_prediction_with_uncertainty(model, metadata, memory_length_TM, X_init, Y_init, t_end=t_end, t_start_plot=prediction_start_time, delta=dt, num_mc_samples=100)

    # 불확실성을 포함한 모든 변수 시각화
    print("\n--- 모든 변수 예측 (불확실성 포함) ---")
    simulate_and_plot_lorenz96_all_variables_prediction_with_uncertainty(model, metadata, memory_length_TM, X_init, Y_init, t_end=t_end, t_start_plot=prediction_start_time, delta=dt, num_mc_samples=100)

    # 불확실성을 포함한 Random IC에 대한 Extrapolation 성능 평가
    print("\n[+] Random IC에 대한 Extrapolation 성능 평가 (불확실성 포함)...")
    evaluate_extrapolation_performance_with_uncertainty(model, metadata, memory_length_TM, num_trials=5, t_end=t_end, t_start_plot=prediction_start_time, delta=dt, num_mc_samples=100)
    