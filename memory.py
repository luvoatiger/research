import os
import sys
import time
import random
import torch

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
        X_list, Y_list = [], []
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
                X_list.append(Z_j_i)
                Y_list.append(z_j_i)

        X = np.array(X_list)
        Y = np.array(Y_list)
        print(f"[Random] X.shape = {X.shape}, Y.shape = {Y.shape}")
        return X, Y

    def create_deterministic_training_dataset(self, all_trajectories):
        X_list, Y_list = [], []
        input_len = self.memory_range_NM + 1
        total_len = self.memory_range_NM + 2

        for traj in all_trajectories:
            max_start = len(traj) - total_len
            if max_start < 1:
                continue
            for start in range(max_start + 1):
                Z_j_i = traj[start:start + input_len].reshape(-1)
                z_j_i = traj[start + input_len]
                X_list.append(Z_j_i)
                Y_list.append(z_j_i)

        X = np.array(X_list)
        Y = np.array(Y_list)
        print(f"[Deterministic] X.shape = {X.shape}, Y.shape = {Y.shape}")
        return X, Y

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


class dAMZ(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(dAMZ, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x):
        pass


# 예시 실행
if __name__ == "__main__":
    # ChaoticSystem 인스턴스 생성
    chaotic_system = ChaoticSystem(eps=0.01)

    # trajectory 생성
    trajectories = chaotic_system.generate_trajectories()
    print(f"생성된 trajectory 수: {len(trajectories)}")
    print(f"각 trajectory shape: {trajectories[0].shape}")  # (101, 3)
    # 시스템 정보 출력
    print("시스템 정보:")
    print(chaotic_system.get_system_info())

    X, Y = chaotic_system.create_training_dataset(trajectories, J0=50)

    # 그래프 시각화로 데이터 확인
    chaotic = ChaoticSystem(dt=0.02, N_T=100, memory_length_TM=1.2)
    simulate_and_plot_x1(chaotic)

    # ResNet 구조로 d-AMZ 학습

    '''
    results_dir = os.path.join(os.getcwd(), "simulated_data")
    with open(f"{results_dir}/metadata.json", "w") as f:
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
    ic_interval = metadata['ic_interval']
    num_ic = metadata['num_ic']
    time_steps_per_ic = metadata['time_steps_per_ic']
    num_batches = metadata['num_batches']
    batch_size = metadata['batch_size']

    memory_length_TM = 100
    memory_range_NM = int(memory_length_TM / dt)

    # 데이터 로드
    data_list = []
    for i in range(1, num_ic + 1):
        X_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"X_batch_{i}.npy"))
        data_list.append([X_data])
    '''

