import os
import sys
import time
import random
import torch

import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp

class ChaoticSystem:
    """
    Chaotic System 클래스 - Eq. (29)를 구현한 카오스 시스템
    """
    
    def __init__(self, eps=0.01):
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

        self.N_T = 100  # trajectory 개수
        self.dt = 0.02
        self.memory_length_TM = 60
        self.memory_range_NM = int(self.memory_length_TM / self.dt)


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
        x0_list = self.sample_initial_conditions(self.N_T)
        all_trajectories = [self.generate_single_trajectory(x0, K, delta) for x0 in x0_list]
        return all_trajectories
    
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
    
