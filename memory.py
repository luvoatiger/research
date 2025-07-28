import os
import sys
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


class ChaoticSystem:
    """
    Chaotic System 클래스 - Eq. (29)를 구현한 카오스 시스템
    """

    def __init__(self, dt = 0.02, N_T = 100, memory_length_TM = 1.2, eps=0.01, selection_mode = 'random'):
        """
        ChaoticSystem 초기화

        Args:
            eps (float): 시스템 파라미터 (기본값: 0.01)
        """
        self.eps = eps

        # 초기 조건 범위 설정
        self.x1_range = (-7.5, 10)
        self.x2_range = (-10, 7.5)
        self.x3_range = (0, 18)
        self.y_range = (-1, 100)

        self.trajectory_num_NT = N_T  # trajectory 개수
        self.sequence_length_Ki = []
        self.dt = dt
        self.memory_length_TM = memory_length_TM

        self.selection_mode = selection_mode
        self.memory_range_NM = int(self.memory_length_TM / self.dt)
        self.sequence_length = self.memory_range_NM + 2

    def system_equations(self, t, x):
        """
        Eq. (29) 정의 - 카오스 시스템의 미분 방정식

        Args:
            t (float): 시간
            x (list): 상태 변수 [x1, x2, x3, y]

        Returns:
            list: 미분값 [dx1, dx2, dx3, dy]
        """
        x1, x2, x3, y = x
        dx1 = -x2 - x3
        dx2 = x1 + (1/5) * x2
        dx3 = 1/5 + y - 5 * x3
        dy = (-y + x1 * x2) / self.eps


        return [dx1, dx2, dx3, dy]

    def set_parameter_ranges(self, x1_range=None, x2_range=None, x3_range=None, y_range=None):
        """
        초기 조건 범위 설정

        Args:
            x1_range (tuple): x1 범위
            x2_range (tuple): x2 범위
            x3_range (tuple): x3 범위
            y_range (tuple): y 범위
        """
        if x1_range is not None:
            self.x1_range = x1_range
        if x2_range is not None:
            self.x2_range = x2_range
        if x3_range is not None:
            self.x3_range = x3_range
        if y_range is not None:
            self.y_range = y_range

    def get_system_info(self):
        """
        시스템 정보 반환

        Returns:
            dict: 시스템 파라미터 정보
        """
        return {
            'eps': self.eps,
            'x1_range': self.x1_range,
            'x2_range': self.x2_range,
            'x3_range': self.x3_range,
            'y_range': self.y_range
        }

    def sample_initial_conditions(self, N):
        """
        초기 조건 범위에서 샘플링

        Args:
            N (int): 샘플 개수

        Returns:
            np.ndarray: 초기 조건들 (N, 4)
        """
        x1 = np.random.uniform(self.x1_range[0], self.x1_range[1], N)
        x2 = np.random.uniform(self.x2_range[0], self.x2_range[1], N)
        x3 = np.random.uniform(self.x3_range[0], self.x3_range[1], N)
        y = np.random.uniform(self.y_range[0], self.y_range[1], N)
        return np.stack([x1, x2, x3, y], axis=1)

    def generate_single_trajectory(self, x0, K=100, delta=0.02):
        """
        단일 trajectory 생성기

        Args:
            x0 (np.ndarray): 초기 조건
            K (int): 시간 스텝 수
            delta (float): 시간 간격

        Returns:
            np.ndarray: trajectory (K+1, 3) - x1, x2, x3만 포함
        """
        t_span = (0, K * delta)
        t_eval = np.linspace(t_span[0], t_span[1], K + 1)
        sol = solve_ivp(
            fun=self.system_equations,
            t_span=t_span,
            y0=x0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )
        z_traj = sol.y[:3, :].T  # x1, x2, x3만 추출

        return z_traj

    def generate_trajectories(self, K=100, delta=0.02, seed=42):
        """
        전체 trajectory 생성기

        Args:
            N_T (int): trajectory 개수
            K (int): 시간 스텝 수
            delta (float): 시간 간격
            seed (int): 랜덤 시드

        Returns:
            list: trajectory들의 리스트
        """
        np.random.seed(seed)
        x0_list = self.sample_initial_conditions(self.trajectory_num_NT)
        all_trajectories = [self.generate_single_trajectory(x0, K, delta) for x0 in x0_list]
        all_trajectories = list(filter(lambda x: len(x) >= self.sequence_length, all_trajectories))

        return all_trajectories

    def create_training_dataset(self, all_trajectories, J0=5):
        if self.selection_mode == 'random':
            return self.create_random_training_dataset(all_trajectories, J0)
        elif self.selection_mode == 'deterministic':
            return self.create_deterministic_training_dataset(all_trajectories)
        else:
            raise ValueError(f"Invalid selection mode: {self.selection_mode}")

    def create_random_training_dataset(self, all_trajectories, J0=5):
        Z_list, z_list = [], []
        input_len = self.memory_range_NM + 1
        total_len = self.memory_range_NM + 2

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

    def create_deterministic_training_dataset(self, all_trajectories):
        Z_list, z_list = [], []
        input_len = self.memory_range_NM + 1
        total_len = self.memory_range_NM + 2

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

def simulate_and_plot_x1(chaotic_system, t_end=400, t_start_plot=200, delta=0.02, initial_condition=None):
    """
    chaotic_system: ChaoticSystem 클래스 인스턴스
    t_end: 적분 끝 시간 (기본: 400)
    t_start_plot: x1을 시각화할 구간의 시작 시간 (기본: 200)
    delta: 시간 간격
    initial_condition: 초기 조건 (None이면 클래스의 샘플러에서 무작위 추출)
    """
    # 초기 조건 설정
    if initial_condition is None:
        initial_condition = chaotic_system.sample_initial_conditions(1)[0]

    # 시간 벡터
    t_eval = np.arange(0, t_end + delta, delta)
    t_span = (0, t_end)

    # 적분 수행
    sol = solve_ivp(
        fun=chaotic_system.system_equations,
        t_span=t_span,
        y0=initial_condition,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-9,
        atol=1e-11
    )

    # 결과 추출
    t_vals = sol.t
    x1_vals = sol.y[0, :]  # x1(t)

    # 시각화 구간 필터링
    mask = (t_vals >= t_start_plot)
    t_plot = t_vals[mask]
    x1_plot = x1_vals[mask]

    # 시각화
    plt.figure(figsize=(10, 4))
    plt.plot(t_plot, x1_plot, label="$x_1(t)$")
    plt.xlabel("Time t")
    plt.ylabel("$x_1(t)$")
    plt.title(f"$x_1(t)$ over time [{t_start_plot}, {t_end}]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


class dAMZ(nn.Module):
    """
    d-AMZ (deep Artificial Memory Zone) 신경망
    논문 Section 3.3의 수식에 따른 구현
    
    Eq. (13): D = d × (n_M + 1)
    Eq. (14): Z = (z_n^T, z_{n-1}^T, ..., z_{n-nM}^T)^T ∈ R^D
    Eq. (15): N(⋅; Θ) : R^D → R^d
    Eq. (16): z^out = [Î + N] (Z^in)
    Eq. (17): z_{n+1} = z_n + N(z_n, z_{n-1}, ..., z_{n-nM}; Θ), n ≥ n_M
    """
    
    def __init__(self, d=3, n_M=60, hidden_dim=30):
        """
        dAMZ 신경망 초기화
        
        Args:
            d (int): 축소된 변수의 차원 (기본값: 3, x1, x2, x3)
            n_M (int): 메모리 항목 수 (memory_length_TM / dt)
            hidden_dim (int): 은닉층 차원
        """
        super(dAMZ, self).__init__()
        
        self.d = d  # 축소된 변수 차원
        self.n_M = n_M  # 메모리 항목 수
        self.D = d * (n_M + 1)  # Eq. (13): D = d × (n_M + 1)
        
        # Î 행렬 정의: Î = [I_d, 0, ..., 0] (d × D)
        # I_d는 (d × d) 단위행렬, n_M개의 (d × d) 영행렬과 연결
        self.I_hat = torch.zeros(d, self.D)
        self.I_hat[:d, :d] = torch.eye(d)  # 첫 번째 d×d 블록만 단위행렬
        
        # 신경망 N(⋅; Θ) 정의: R^D → R^d
        self.neural_network = nn.Sequential(
            nn.Linear(self.D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, Z_in):
        """
        Forward pass - Eq. (16)과 Eq. (17) 구현
        
        Args:
            Z_in (torch.Tensor): 입력 텐서 [batch_size, D]
                                Z = (z_n^T, z_{n-1}^T, ..., z_{n-nM}^T)^T
        
        Returns:
            torch.Tensor: 출력 [batch_size, d] - z_{n+1}
        """
        batch_size = Z_in.shape[0]
        
        # Î 행렬을 배치 크기에 맞게 확장
        I_hat_batch = self.I_hat.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, d, D]
        
        # Z_in을 배치 차원에서 행렬로 변환
        Z_in_matrix = Z_in.unsqueeze(2)  # [batch_size, D, 1]
        
        # Î × Z_in 계산 (z_n 추출)
        z_n = torch.bmm(I_hat_batch, Z_in_matrix).squeeze(2)  # [batch_size, d]
        
        # 신경망 N(Z_in) 계산
        N_output = self.neural_network(Z_in)  # [batch_size, d]
        
        # Eq. (16): z^out = [Î + N] (Z^in) = z_n + N(Z_in)
        # Eq. (17): z_{n+1} = z_n + N(z_n, z_{n-1}, ..., z_{n-nM}; Θ)
        z_out = z_n + N_output
        
        return z_out
    
    def get_memory_structure_info(self):
        """메모리 구조 정보 반환"""
        return {
            'd': self.d,
            'n_M': self.n_M,
            'D': self.D,
            'input_dim': self.D,
            'output_dim': self.d
        }


def train_model(model, Z, z, epochs=2000, lr=1e-3):
    """
    dAMZ 모델 학습 함수
    
    Args:
        model: dAMZ 모델
        Z: 입력 데이터 [batch_size, D]
        z: 타겟 데이터 [batch_size, d]
        epochs: 학습 에포크 수
        lr: 학습률
    """
    # 데이터를 텐서로 변환
    Z_tensor = torch.FloatTensor(Z)
    z_tensor = torch.FloatTensor(z)
    
    # 손실 함수와 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 학습
    model.train()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 순전파
        outputs = model(Z_tensor)
        loss = criterion(outputs, z_tensor)
        
        # 역전파
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    print(f'Training completed. Final loss: {losses[-1]:.6f}')
    return losses


def simulate_and_plot_x1_prediction(model, chaotic_system, t_end=400, t_start_plot=200, delta=0.02):
    """
    모델 예측과 실제 시뮬레이션을 비교하여 x1을 시각화하는 함수
    
    Args:
        model: 학습된 dAMZ 모델
        chaotic_system: ChaoticSystem 인스턴스
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
    """
    # 초기 조건 설정
    initial_condition = chaotic_system.sample_initial_conditions(1)[0]
    
    # 실제 시뮬레이션
    t_eval = np.arange(0, t_end + delta, delta)
    t_span = (0, t_end)
    
    sol = solve_ivp(
        fun=chaotic_system.system_equations,
        t_span=t_span,
        y0=initial_condition,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-9,
        atol=1e-11
    )
    
    t_vals = sol.t
    x1_true = sol.y[0, :]  # 실제 x1 값
    
    # 모델 예측
    model.eval()
    n_M = chaotic_system.memory_range_NM
    
    # 메모리 항목들을 포함한 입력 데이터 준비
    z_traj = sol.y[:3, :].T  # x1, x2, x3만 추출
    
    x1_pred = []
    with torch.no_grad():
        for i in range(n_M + 1, len(z_traj)):
            # Z = (z_n^T, z_{n-1}^T, ..., z_{n-nM}^T)^T
            Z_input = z_traj[i-n_M-1:i].reshape(-1)  # [D]
            Z_tensor = torch.FloatTensor(Z_input).unsqueeze(0)  # [1, D]
            
            # 예측
            z_pred = model(Z_tensor)
            x1_pred.append(z_pred[0, 0].item())  # x1 예측값
    
    # 시각화
    t_pred = t_vals[n_M + 1:]
    mask = (t_pred >= t_start_plot)
    t_plot = t_pred[mask]
    x1_true_plot = x1_true[n_M + 1:][mask]
    x1_pred_plot = np.array(x1_pred)[mask]
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_plot, x1_true_plot, 'b-', label='True $x_1(t)$', linewidth=2)
    plt.plot(t_plot, x1_pred_plot, 'r--', label='Pred $x_1(t)$', linewidth=2)
    plt.xlabel('time t')
    plt.ylabel('$x_1(t)$')
    plt.title(f'dAMZ prediction vs real simulation [{t_start_plot}, {t_end}]')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # MSE 계산
    mse = np.mean((x1_true_plot - x1_pred_plot) ** 2)
    print(f'예측 MSE: {mse:.6f}')


if __name__ == "__main__":
    # 설정
    dt = 0.02
    memory_length_TM = 1.2
    hidden_dim = 30
    epochs = 2000
    J0 = 50

    # ChaoticSystem 인스턴스 생성
    chaotic_system = ChaoticSystem(
        dt=dt,
        N_T=100,
        memory_length_TM=memory_length_TM,
        eps=0.01,
        selection_mode='random'
    )

    # Trajectory 생성
    trajectories = chaotic_system.generate_trajectories(K=100, delta=dt)
    print(f"생성된 trajectory 수: {len(trajectories)}")
    print(f"각 trajectory shape: {trajectories[0].shape}")

    # 학습 데이터 생성
    Z, z = chaotic_system.create_training_dataset(trajectories, J0=J0)

    # 모델 생성 - 새로운 dAMZ 구조에 맞게 수정
    d = 3  # 축소된 변수 차원 (x1, x2, x3)
    n_M = chaotic_system.memory_range_NM  # 메모리 항목 수
    model = dAMZ(d=d, n_M=n_M, hidden_dim=hidden_dim)
    
    # 모델 구조 정보 출력
    model_info = model.get_memory_structure_info()
    print(f"\n[+] dAMZ 모델 구조:")
    print(f"  - 축소된 변수 차원 (d): {model_info['d']}")
    print(f"  - 메모리 항목 수 (n_M): {model_info['n_M']}")
    print(f"  - 입력 차원 (D): {model_info['D']}")
    print(f"  - 출력 차원: {model_info['output_dim']}")
    print(f"  - 실제 입력 데이터 shape: {Z.shape}")
    print(f"  - 실제 출력 데이터 shape: {z.shape}")

    # 학습
    print("\n[+] Training model...")
    train_model(model, Z, z, epochs=epochs, lr=1e-3)

    # 시뮬레이션 vs 예측 시각화 (더 짧은 구간으로 테스트)
    print("\n[+] Simulating and plotting prediction vs ground truth...")
    simulate_and_plot_x1_prediction(model, chaotic_system, t_end=400, t_start_plot=200, delta=0.02)

