import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from evaluation import *


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

        # LSTM 네트워크 N(⋅; Θ) 정의: R^D → R^d (memory term만 학습)
        # 입력을 시퀀스로 재구성하여 LSTM에 전달
        self.lstm = nn.LSTM(
            input_size=d,  # 각 시점의 입력 차원
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if 2 > 1 else 0,
            bidirectional=False
        )
        
        # LSTM 출력을 최종 차원으로 매핑
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, d)
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
        elif isinstance(module, nn.LSTM):
            # LSTM 가중치 초기화
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param, gain=0.01)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

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
            self.lstm.train()
            self.output_layer.train()
        else:
            self.lstm.eval()
            self.output_layer.eval()
        
        # LSTM을 위한 입력 재구성: [batch_size, D] -> [batch_size, n_M+1, d]
        # Z_in은 [batch_size, D] 형태이고, D = d * (n_M + 1)
        Z_in_reshaped = Z_in.view(batch_size, self.n_M + 1, self.d)
        
        # LSTM 처리
        lstm_out, _ = self.lstm(Z_in_reshaped)
        
        # 마지막 시점의 출력만 사용 (가장 최근 정보)
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # 최종 출력층을 통한 매핑
        memory_term = self.output_layer(last_output)

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
            'integration_method': 'Euler',
            'neural_network_type': 'LSTM',
            'lstm_layers': 2,
            'lstm_hidden_dim': self.lstm.hidden_size
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


def _rollout_k_steps_with_model(model, batch_Z_flat, k_steps):
    """
    Hist_Deterministic의 stepper와 동일한 아이디어:
    - 입력: [B, D] (히스토리: '오래됨→최근' 순서로 평탄화)
    - k_steps번 자기 예측을 히스토리에 push 하며 진행
    - 반환: 마지막(=k_steps번째) 예측 z_{t+k}
    """
    model.train()  # dropout/BN 일관성 (학습 중)
    B, D = batch_Z_flat.shape
    d = model.d
    nM = model.n_M

    # [B, n_M+1, d]로 복원 (오래됨→최근)
    hist = batch_Z_flat.view(B, nM + 1, d)

    pred_k = None
    for _ in range(k_steps):
        Z_now = hist.view(B, -1)                # [B, D]
        pred_k = model(Z_now, enable_dropout=True)  # [B, d]
        # 히스토리 업데이트: 가장 오래된 프레임 제거, 예측을 "최근" 프레임으로 append
        hist = torch.cat([hist[:, 1:, :], pred_k.unsqueeze(1)], dim=1)

    return pred_k  # [B, d]


def create_transfer_training_dataset(all_trajectories, memory_range_NM, n_fut_transfer, selection_mode='random', J0=5):
    """
    1-step 학습과 달리, 타겟을 '현재 히스토리의 마지막 시점'으로부터 n_fut_transfer 스텝 뒤로 둔다.
    (Hist_Deterministic의 Xt_transfer / Xtpdt_transfer 구성과 동일한 오프셋)
    """
    Z_list, z_future_list = [], []
    input_len = memory_range_NM + 1
    total_len = input_len + n_fut_transfer  # 히스토리 + n_fut 앞 타겟 1개

    choose_starts = []
    for traj in all_trajectories:
        max_start = len(traj) - total_len
        if max_start < 0:  # 샘플이 모자라면 skip
            continue
        if selection_mode == 'random':
            J_i = min(J0, max_start + 1)
            starts = np.random.choice(max_start + 1, J_i, replace=False)
        elif selection_mode == 'deterministic':
            starts = np.arange(max_start + 1)
        else:
            raise ValueError(f"Invalid selection mode: {selection_mode}")
        for start in starts:
            # 히스토리: [start : start+input_len)  (오래됨→최근)
            hist = traj[start:start + input_len].reshape(-1)
            # 타겟: 히스토리 마지막 프레임 기준 n_fut_transfer 스텝 뒤
            target_idx = start + input_len - 1 + n_fut_transfer
            z_future = traj[target_idx]
            Z_list.append(hist)
            z_future_list.append(z_future)

    Z_tf = np.array(Z_list)
    z_tf = np.array(z_future_list)
    print(f"[Transfer] Z_tf.shape = {Z_tf.shape}, z_tf.shape = {z_tf.shape} (n_fut={n_fut_transfer})")
    return Z_tf, z_tf


def train_model_transfer(
    model, Z_tf, z_tf, n_fut_transfer=5, epochs=30000, lr=1e-4, batch_size=256,
    normalize=True, clip_grad_norm=0.5, lambda_horizon_sum=0.0
):
    """
    Hist_Deterministic의 'transfer' 단계(PyTorch 버전).
    - 입력 히스토리를 자기-되먹임으로 n_fut_transfer번 굴려 최종 z_{t+n}을 맞춤
    - lambda_horizon_sum>0이면, 중간 호라이즌까지의 합산 손실(가중치)도 부여 가능
    """
    # 정규화(선택)
    if normalize:
        Z_mean = np.mean(Z_tf, axis=0)
        Z_std  = np.std(Z_tf, axis=0); Z_std = np.where(Z_std == 0, 1.0, Z_std)
        z_mean = np.mean(z_tf,  axis=0)
        z_std  = np.std(z_tf,  axis=0); z_std = np.where(z_std == 0, 1.0, z_std)
        Z_t = torch.tensor((Z_tf - Z_mean) / Z_std, dtype=torch.float32)
        z_t = torch.tensor((z_tf - z_mean) / z_std, dtype=torch.float32)
    else:
        Z_t = torch.tensor(Z_tf, dtype=torch.float32)
        z_t = torch.tensor(z_tf, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(Z_t, z_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)

    model.train()
    losses = []
    best = float('inf'); patience=0; max_patience=200

    for ep in range(epochs):
        ep_loss = 0.0; nb = 0
        for batch_Z, batch_z_future in loader:
            optimizer.zero_grad()

            # 최종 시점 loss
            z_pred_k = _rollout_k_steps_with_model(model, batch_Z, n_fut_transfer)   # [B, d]
            loss = criterion(z_pred_k, batch_z_future)

            # (옵션) 멀티-호라이즌 합산 손실: lambda_horizon_sum>0이면 켬
            if lambda_horizon_sum > 0.0:
                # 1..(n_fut_transfer-1)까지의 예측도 누적(작을수록 가중치↑)
                B = batch_Z.shape[0]
                d = model.d; nM = model.n_M
                hist = batch_Z.view(B, nM + 1, d)
                gamma = 0.95
                acc_loss = 0.0; wsum = 0.0
                for k in range(1, n_fut_transfer):
                    Z_now = hist.view(B, -1)
                    z_k = model(Z_now, enable_dropout=True)
                    hist = torch.cat([hist[:, 1:, :], z_k.unsqueeze(1)], dim=1)
                    acc_loss = acc_loss + (gamma**(k-1)) * criterion(z_k, batch_z_future)  # proxy: 같은 타깃으로 끌어당김
                    wsum += (gamma**(k-1))
                loss = loss + lambda_horizon_sum * (acc_loss / max(wsum, 1e-8))

            # backward
            loss.backward()
            if clip_grad_norm and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()

            ep_loss += loss.item(); nb += 1

        ep_loss /= nb; losses.append(ep_loss)
        scheduler.step(ep_loss)

        if ep_loss < best - 1e-8:
            best = ep_loss; patience = 0
        else:
            patience += 1

        if (ep + 1) % 200 == 0:
            print(f"[Transfer] Epoch {ep+1}/{epochs} | Loss(final@+{n_fut_transfer}): {ep_loss:.6f} | LR {optimizer.param_groups[0]['lr']:.2e}")

        if patience >= max_patience:
            print(f"[Transfer] Early stopping at epoch {ep+1} (best={best:.6f})")
            break

    print(f"[Transfer] Done. Final loss: {losses[-1]:.6f}")
    return losses


if __name__ == "__main__":
    # 설정
    hidden_dim = 30
    epochs = 15000
    J0 = 50
    # 기존에 생성한 Lorenz 96 system dataset 이용
    try:
        print("\n[+] Loading Lorenz 96 system dataset...")
        results_dir = os.path.join(os.getcwd(), "simulated_data")
        with open(os.path.join(results_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error loading metadata.json: {e}")
        print("Using default metadata values...")
        metadata = {}

    K = metadata.get('K', 8)
    J = metadata.get('J', 32)
    F = metadata.get('F', 15.0)
    h = metadata.get('h', 1.0)
    b = metadata.get('b', 10.0)
    c = metadata.get('c', 10.0)
    dt = metadata.get('dt', 0.005)
    si = metadata.get('si', 0.005)
    spinup_time = metadata.get('spinup_time', 3)
    forecast_time = metadata.get('forecast_time', 10)
    num_ic = metadata.get('num_ic', 300)
    
    # metadata 딕셔너리에 기본값들을 추가하여 evaluation 함수들이 사용할 수 있도록 함
    metadata['K'] = K
    metadata['J'] = J
    metadata['F'] = F
    metadata['h'] = h
    metadata['b'] = b
    metadata['c'] = c
    metadata['dt'] = dt

    # 메모리 길이를 더 작게 설정하여 예측 시작점을 앞당김
    memory_length_TM = 0.02
    memory_range_NM = int(memory_length_TM / dt)
    
    # 데이터 로드 및 전처리
    trajectories = []
    for i in range(1, min(num_ic + 1, 51)):  # 처음 50개만 사용하여 메모리 절약
        try:
            X_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"X_batch_coupled_{i}.npy"))
            # X_data shape: (1, time_steps, 8) -> (time_steps, 8)로 변환
            trajectory = X_data[0]  # 첫 번째 배치만 사용
#            trajectory = trajectory[::2,:]
#            trajectory = trajectory + 0.003*np.std(trajectory, axis=0)*torch.randn(trajectory.shape, device=torch.device('cpu')).numpy()
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
    
    Z, z = create_training_dataset(trajectories, memory_range_NM, selection_mode='random', J0=J0)
    
    # 데이터 품질 체크
    print(f"Z 데이터 통계: min={Z.min():.4f}, max={Z.max():.4f}, mean={Z.mean():.4f}, std={Z.std():.4f}")
    print(f"z 데이터 통계: min={z.min():.4f}, max={z.max():.4f}, mean={z.mean():.4f}, std={z.std():.4f}")
    
    if np.any(np.isnan(Z)) or np.any(np.isnan(z)):
        print("오류: 학습 데이터에 NaN 값이 포함되어 있습니다!")
        sys.exit(1)

    # Lorenz 96 시스템에 맞는 모델 생성 (Monte Carlo Dropout 포함)
    reduced_order_d = K  # Lorenz 96 시스템의 변수 수
    dropout_rate = 0.1  # dropout 비율
    print(f"모델 생성 시작: reduced_order_d={reduced_order_d}, n_M={memory_range_NM}, hidden_dim={hidden_dim}, dropout_rate={dropout_rate}")
    model = dAMZ(d=reduced_order_d, n_M=memory_range_NM, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
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
    
        # === (1) 1-step 학습 완료 후 ===
    print("\n[+] 1-step training completed.")

    '''
    # === (2) Transfer 데이터셋 만들고, n_fut-step ahead로 미세조정 ===
    n_fut_transfer = 5   # Hist_Deterministic.py의 기본 예
    Z_tf, z_tf = create_transfer_training_dataset(
        trajectories, memory_range_NM, n_fut_transfer,
        selection_mode='random', J0=J0
    )
    print(f"Transfer Z/z stats: Z[{Z_tf.min():.4f},{Z_tf.max():.4f}], z[{z_tf.min():.4f},{z_tf.max():.4f}]")

    print("\n[+] Starting transfer learning (self-feeding to +%d steps)..." % n_fut_transfer)
    transfer_losses = train_model_transfer(
        model, Z_tf, z_tf,
        n_fut_transfer=n_fut_transfer,
        epochs=30000,         # Hist 코드(3만 스텝)와 유사
        lr=1e-4,              # Hist 코드 1e-4
        batch_size=256,
        normalize=True,       # 1-step 때와 동일 정책 유지
        clip_grad_norm=0.5,
        lambda_horizon_sum=0.0 # =0이면 Hist와 동일(마지막 시점만 loss); >0이면 멀티-호라이즌 가중
    )
    '''

    # 이후 시뮬레이션/평가 코드는 그대로 사용 (모델 파라미터가 업데이트된 상태)

    print("\n[+] Lorenz 96 시스템 학습 완료!")
        
    batch_num = np.random.randint(1, 300)

    # 예측 시각화
    print("\n[+] Simulating and plotting prediction vs ground truth...")
    
    # 예측 시작 시간 계산
    prediction_start_time = (memory_range_NM + 1) * dt  # 실제 예측이 시작되는 시간
    
    # 모든 변수 시각화
    t_end = 10
    print("\n--- 모든 변수 예측 ---")
    simulate_and_plot_lorenz96_all_variables_prediction(model, metadata, memory_length_TM, trajectory_file=f"X_batch_coupled_{batch_num}.npy", t_end=t_end, t_start_plot=prediction_start_time, delta=dt)