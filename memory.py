import numpy as np
import traceback
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import traceback

from torch.utils.data import DataLoader, Dataset
from torchdde import integrate, AdaptiveStepSizeController, RK4, Dopri5
from scipy.integrate import solve_ivp
from tqdm import tqdm

import os
from sklearn.preprocessing import StandardScaler
from IPython.display import clear_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DataScaler:
    """데이터 정규화를 위한 스케일러 클래스"""
    
    def __init__(self, method='standard'):
        """
        Args:
            method (str): 정규화 방법 ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.scaler = None
        self.is_fitted = False
        
    def fit(self, data):
        """데이터에 맞춰 스케일러 학습"""
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        elif self.method == 'robust':
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
            
        # 데이터 형태에 따라 적절히 reshape
        if data.ndim == 3:  # (batch, time, features)
            data_reshaped = data.reshape(-1, data.shape[-1])
        else:
            data_reshaped = data
            
        self.scaler.fit(data_reshaped)
        self.is_fitted = True
        
    def transform(self, data):
        """데이터 변환"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
            
        original_shape = data.shape
        if data.ndim == 3:  # (batch, time, features)
            data_reshaped = data.reshape(-1, data.shape[-1])
            transformed = self.scaler.transform(data_reshaped)
            return transformed.reshape(original_shape)
        else:
            return self.scaler.transform(data)
            
    def inverse_transform(self, data):
        """역변환"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
            
        original_shape = data.shape
        if data.ndim == 3:  # (batch, time, features)
            data_reshaped = data.reshape(-1, data.shape[-1])
            inverse_transformed = self.scaler.inverse_transform(data_reshaped)
            return inverse_transformed.reshape(original_shape)
        else:
            return self.scaler.inverse_transform(data)
            
    def get_scaling_info(self):
        """스케일링 정보 반환"""
        if not self.is_fitted:
            return None
            
        if self.method == 'standard':
            return {
                'mean': self.scaler.mean_,
                'scale': self.scaler.scale_,
                'method': self.method
            }
        elif self.method == 'minmax':
            return {
                'min': self.scaler.min_,
                'scale': self.scaler.scale_,
                'method': self.method
            }
        elif self.method == 'robust':
            return {
                'center': self.scaler.center_,
                'scale': self.scaler.scale_,
                'method': self.method
            }


class ScaledLorenz96Dataset(Dataset):
    """스케일링이 적용된 Lorenz 96 데이터셋"""
    
    def __init__(self, time_axis, trajectory_value, scaler=None):
        self.ts = time_axis
        self.ys = trajectory_value
        self.scaler = scaler
        
        # 스케일링 적용
        if self.scaler is not None:
            self.ys_scaled = self.scaler.transform(self.ys)
        else:
            self.ys_scaled = self.ys

    def __getitem__(self, index):
        ts = torch.tensor(self.ts[index], dtype=torch.float32)
        if self.scaler is not None:
            ys = torch.tensor(self.ys_scaled[index], dtype=torch.float32)
        else:
            ys = torch.tensor(self.ys[index], dtype=torch.float32)
        return ts, ys

    def __len__(self):
        return self.ys.shape[0]


def lorenz96_two_scale_rhs(t, state, K=8, J=32, F=10, h=1, b=10, c=10):
    """
    Two-scale Lorenz 96 시스템의 우변 함수
    state: [X_1, ..., X_K, Y_1, ..., Y_{J*K}]
    """
    X = state[:K]
    Y = state[K:].reshape(K, J)
    dXdt = np.zeros(K)
    dYdt = np.zeros((K, J))
    # X 방정식
    for k in range(K):
        dXdt[k] = (
            (X[(k+1)%K] - X[k-2]) * X[k-1]
            - X[k]
            + F
            - (h * c / b) * np.sum(Y[k])
        )
    # Y 방정식
    for k in range(K):
        for j in range(J):
            dYdt[k, j] = (
                -c * b * Y[k, j]
                + (c / b) * (Y[k, (j+1)%J] - Y[k, j-2]) * Y[k, j-1]
                + (h * c / b) * X[k]
            )
    return np.concatenate([dXdt, dYdt.reshape(-1)])


class MZMemoryDDE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, delay_Tau):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.delay_Tau = delay_Tau

        # 입력 차원: 현재 상태 + 과거 상태들
        total_input_dim = input_dim * (1 + delay_Tau)

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t, z, history):
        """
        Args:
            t: 현재 시간 (torchdde에서 자동으로 전달)
            z: 현재 상태 [batch_size, input_dim] (2차원) 또는 [input_dim] (1차원)
            history: 과거 상태들의 리스트 [x(t-lag_M), x(t-2*lag_M), ..., x(t-delay_Tau*lag_M)]
        """
        combined = torch.cat([z, *history], dim=-1)
        output = self.mlp(combined)

        return output


class L96MZNetwork(nn.Module):
    def __init__(self, K, F, delay_Tau, lag_M, hidden_dim, dt):
        super().__init__()
        self.delay_Tau = delay_Tau
        self.lag_M = lag_M
        self.K = K
        self.F = F
        self.dt = dt
        # DDE 지연 시간 목록 (초 단위): [lag_M*dt, 2*lag_M*dt, ..., delay_Tau*lag_M*dt]
        self.delays = torch.tensor([float(i * lag_M * dt) for i in range(1, delay_Tau + 1)], dtype=torch.float32)
        self.memory_dde = MZMemoryDDE(K, hidden_dim, K, delay_Tau)

    def lorenz_96_markov_term(self, z_n):
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


    def forward(self, t, z, func_args, *, history):
        markov_term = self.lorenz_96_markov_term(z)
        memory_term = self.memory_dde(t, z, history)

        return markov_term + memory_term


def to_np(x):
    return x.detach().cpu().numpy()


def plot_trajectories(obs=None, times=None, trajs=None, save=None, figsize=(16, 8), save_path=None):
    plt.figure(figsize=figsize)
    plt.plot(to_np(times), to_np(obs)[:, 0], label='real')
    plt.plot(to_np(times), to_np(trajs)[:, 0], label='predicted')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def train_neural_dde(model, train_loader, optimizer, criterion, device, dt=0.005, delay_Tau=2, lag_M=2):
    """
    Neural DDE 모델 학습 (배치 단위 처리)
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    count = 0
    for batch_idx, (time_series, trajectory_batch) in enumerate(train_loader):
        delays_vector = torch.as_tensor([(i + 1) * lag_M *dt for i in range(delay_Tau)],
                                    dtype=time_series.dtype, device=device)
        # trajectory_batch: (B, time_steps, K) 형태
        B, time_steps, K = trajectory_batch.shape


       # ---- 딜레이/시작점 설정 ----
        max_delay_samples = delay_Tau * lag_M                # τ_max = max_delay_samples * dt
        t0_idx   = max_delay_samples
        hist_times = time_series[0][:t0_idx+1]                       # (t in [t0-τ_max, t0])
        hist_vals  = trajectory_batch[:, :t0_idx+1, :].detach()        # (B, t0_idx+1, K), 상수 history

        # 배치 단위로 데이터를 device로 이동
        trajectory_batch = trajectory_batch.to(device, dtype=time_series.dtype)  # (B, time_steps, K)

        optimizer.zero_grad()
        # ---- 선형보간 history_fn: (B,K) 반환 ----
        def history_fn(tq: torch.Tensor):
            # tq는 scalar; [hist_times[0], hist_times[-1]]로 clamp
            tqc = torch.clamp(tq, hist_times[0], hist_times[-1])
            # 우측 구간 index
            i1 = torch.searchsorted(hist_times, tqc)
            i1 = torch.clamp(i1, 1, hist_times.numel() - 1)
            i0 = i1 - 1
            t_left  = hist_times[i0]
            t_right = hist_times[i1]
            w = (tqc - t_left) / (t_right - t_left + 1e-12)       # scalar
            y0 = hist_vals[:, i0, :]                              # (B,K)
            y1 = hist_vals[:, i1, :]                              # (B,K)
            return (1.0 - w) * y0 + w * y1                        # (B,K)

        # 배치 단위로 integrate 호출
        solution = integrate(
            func=model,
            solver=RK4(),
            t0=time_series[0, 0],
            t1=time_series[0, -1],
            ts=time_series[0],
            y0=history_fn,
            func_args=None,
            stepsize_controller=AdaptiveStepSizeController(rtol=1e-6, atol=1e-8),
            dt0=time_series[0, 1] - time_series[0, 0],
            delays=delays_vector,
            discretize_then_optimize=False)

        assert solution.shape[0] == trajectory_batch.shape[0], f"B mismatch: {solution.shape} vs {trajectory_batch.shape}"
        assert solution.shape[2] == trajectory_batch.shape[2], f"K mismatch: {solution.shape} vs {trajectory_batch.shape}"
        assert solution.shape[1] == trajectory_batch.shape[1], (
            f"time length mismatch: pred T={solution.shape[1]} vs target T={trajectory_batch.shape[1]}. "
            f"Dataset/target must be (B, T, K). Check your dataset & history_fn."
        )
        loss = criterion(solution, trajectory_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_samples += B

        if count == 0 or count % 10 == 0:
            plot_trajectories(obs=trajectory_batch[0], times=time_series[0], trajs=solution[0], 
                                save_path=os.path.join(os.getcwd(), f"{count}.png"))
        count += 1
    return total_loss / num_samples if num_samples > 0 else float("inf")


def evaluate_neural_dde(model, test_loader, criterion, device, dt=0.005, delay_Tau=2, lag_M=2):
    """
    Neural DDE 모델 평가 (배치 단위 처리)
    """
    model.eval()
    total_loss = 0.0
    num_samples = 0


    with torch.no_grad():
        for batch_idx, (time_series, trajectory_batch) in enumerate(test_loader):
            delays_vector = torch.as_tensor([(i + 1) * lag_M *dt for i in range(delay_Tau)],
                                        dtype=time_series.dtype, device=device)
            # trajectory_batch: (B, time_steps, K) 형태
            B, time_steps, K = trajectory_batch.shape

            # 배치 단위로 데이터를 device로 이동
            trajectory_batch = trajectory_batch.to(device, dtype=torch.float32)  # (B, time_steps, K)

            # delay_Tau=2, lag_M=2일 때 DDE의 초기조건 인덱스는 0
            history_indices = [i * lag_M for i in range(delay_Tau + 1)] # 예: [0, 2, 4]
            history_fn = lambda t: trajectory_batch[:, 0, :].detach()
            # 배치 단위로 integrate 호출
            solution = integrate(
                func=model,
                solver=RK4(),
                t0=time_series[0, 0],
                t1=time_series[0, -1],
                ts=time_series[0],
                y0=history_fn,
                func_args=None,
                stepsize_controller=AdaptiveStepSizeController(rtol=1e-6, atol=1e-8),
                dt0=time_series[0, 1] - time_series[0, 0],
                delays=delays_vector,
                discretize_then_optimize=False)
            # ✅ 방어적 점검
            assert solution.shape[0] == trajectory_batch.shape[0], f"B mismatch: {solution.shape} vs {trajectory_batch.shape}"
            assert solution.shape[2] == trajectory_batch.shape[2], f"K mismatch: {solution.shape} vs {trajectory_batch.shape}"
            assert solution.shape[1] == trajectory_batch.shape[1], (
                f"time length mismatch: pred T={solution.shape[1]} vs target T={trajectory_batch.shape[1]}. "
                f"Dataset/target must be (B, T, K). Check your dataset & history_fn."
            )
            loss = criterion(solution, trajectory_batch)
            total_loss += loss.item()
            num_samples += B

    return total_loss / num_samples if num_samples > 0 else float("inf")

def train_and_save_model(dt, batch_size, lr, max_epoch, delay_Tau, lag_M, K, F, hidden_dim, scaling_method):
    """학습 및 모델 저장 함수"""
    print(f"\n=== 학습 및 모델 저장 시작 ===")
    
    # 데이터 생성
    print("\n[+] Trajectory 생성 중...")

    # 여러 개의 trajectory 생성 (여기서는 5개 생성)
    num_trajectories = 1
    trajectories = []
    time_list = []
    for i in range(1, 1+num_trajectories):
        # 각각 다른 초기값으로 trajectory 생성
        traj_i = np.load(os.path.join(os.getcwd(), "simulated_data", f"X_batch_coupled_{i}.npy"))[0]
        t_i = np.load(os.path.join(os.getcwd(), "simulated_data", f"t_batch_coupled_{i}.npy"))[0]
        if t_i[0] != 0:
            t_i = t_i - t_i[0]
        # traj_i의 실제 shape에 따라 안전 변환
        # 기대 형태: 최종적으로 X_traj_i는 (T, K)
        if traj_i.ndim == 2:
            if traj_i.shape[0] == K:         # (K, T)인 경우
                X_traj_i = traj_i[:K, :].T   # -> (T, K)
            elif traj_i.shape[1] == K:       # (T, K)인 경우
                X_traj_i = traj_i[:, :K]     # -> (T, K)
            else:
                raise ValueError(f"traj_i shape={traj_i.shape}, cannot infer (T,K).")
        else:
            raise ValueError(f"traj_i ndim={traj_i.ndim}, expected 2D array.")
        print(X_traj_i.shape)
        print(t_i.shape)
        trajectories.append(X_traj_i)
        time_list.append(t_i)

    # 모든 trajectory를 하나의 배열로 합치기
    trajectories = np.array(trajectories)  # [num_trajectories, time_steps, K]
    time_list = np.array(time_list)
    print(f"Trajectories shape: {trajectories.shape}")
    
    # 데이터 스케일링 적용
    print(f"\n[+] 데이터 스케일링 적용 중...")
    scaler = DataScaler(method=scaling_method)
    scaler.fit(trajectories)
    
    # 스케일링 정보 출력
    scaling_info = scaler.get_scaling_info()
    print(f"스케일링 방법: {scaling_info['method']}")
    print(f"평균: {scaling_info['mean']}")
    print(f"표준편차: {scaling_info['scale']}")
    
    # 스케일링된 데이터 확인
    trajectories_scaled = scaler.transform(trajectories)
    print(f"원본 데이터 범위: [{np.min(trajectories):.3f}, {np.max(trajectories):.3f}]")
    print(f"스케일링된 데이터 범위: [{np.min(trajectories_scaled):.3f}, {np.max(trajectories_scaled):.3f}]")

    # 데이터셋 생성 (스케일링 적용)
    print(f"\n[+] 데이터셋 생성 중...")
    train_dataset = ScaledLorenz96Dataset(time_list, trajectories, scaler)
    print(f"데이터셋 크기: {len(train_dataset)}")

    # --- (2) DataLoader(선택적) 성능 옵션 ---
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == "cuda"), num_workers=2, drop_last=False
    )

    # 모델 초기화
    print(f"\n[+] 모델 초기화 중...")
    model = L96MZNetwork(K, F, delay_Tau, lag_M, hidden_dim, 0.005).to(device)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")

    # 손실 함수 및 옵티마이저
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 학습 루프
    print(f"\n[+] 학습 시작...")
    print(f"AdaptiveStepSizeController 설정:")
    print(f"  - rtol: 1e-6 (상대 오차 허용치)")
    print(f"  - atol: 1e-8 (절대 오차 허용치)")

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(max_epoch)):
        start_time = time.time()

        # 학습
        train_loss = train_neural_dde(model, train_loader, optimizer, criterion, device, dt, delay_Tau, lag_M)

        # 검증
#        val_loss = evaluate_neural_dde(model, val_loader, criterion, device, dt, delay_Tau, lag_M)

        # 학습률 조정
#        scheduler.step(val_loss)

        # 결과 저장
        train_losses.append(train_loss)
#        val_losses.append(val_loss)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{max_epoch}: "
              f"Train Loss = {train_loss:.6f}, "
              f"Time = {epoch_time:.2f}s")

 
    # 학습 결과 시각화
    print(f"\n[+] 학습 결과 시각화...")
    print(f"시각화할 데이터 - train_losses: {train_losses}")
    print(f"시각화할 데이터 - val_losses: {val_losses}")

    if len(train_losses) == 0 or len(val_losses) == 0:
        print("경고: loss 데이터가 없습니다. 그래프를 그릴 수 없습니다.")
    else:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        epochs = list(range(1, len(train_losses) + 1))  # 1부터 시작하는 epoch 번호
        plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.xlim(0.5, len(train_losses) + 0.5)  # X축 범위 명시적 설정

        plt.subplot(1, 2, 2)
        # 최근 20개 데이터만 표시 (데이터가 20개 미만이면 전체 표시)
        recent_train = train_losses[-20:] if len(train_losses) >= 20 else train_losses
        recent_val = val_losses[-20:] if len(val_losses) >= 20 else val_losses

        # 최근 데이터에 대한 epoch 번호 계산
        if len(train_losses) >= 20:
            recent_epochs = list(range(len(train_losses) - 19, len(train_losses) + 1))
        else:
            recent_epochs = list(range(1, len(train_losses) + 1))

        plt.plot(recent_epochs, recent_train, 'b-', label=f'Train Loss (Last {len(recent_train)})', linewidth=2)
        plt.plot(recent_epochs, recent_val, 'r-', label=f'Validation Loss (Last {len(recent_val)})', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Recent Training Progress')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        plt.xlim(min(recent_epochs) - 0.5, max(recent_epochs) + 0.5)  # X축 범위 명시적 설정

        plt.tight_layout()
        plt.show()

    print(f"\n=== 학습 완료 ===")
    print(f"최종 학습 손실: {train_losses[-1]:.6f}")
    print(f"최종 검증 손실: {val_losses[-1]:.6f}")

    # 모델 저장 (스케일러 정보 포함)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'hyperparameters': {
            'delay_Tau': delay_Tau,
            'lag_M': lag_M,
            'K': K,
            'F': F,
            'hidden_dim': hidden_dim
        },
        'scaler_info': scaling_info
    }, 'lorenz96_neural_dde_model.pth')

    print(f"모델이 'lorenz96_neural_dde_model.pth'에 저장되었습니다.")
    print(f"스케일러 정보도 함께 저장되었습니다.")


def load_and_inference_model(dt, delay_Tau, lag_M, K, F, hidden_dim):
    """모델 로딩 및 추론 함수"""
    print(f"\n=== 모델 로딩 및 추론 시작 ===")
    
    # 저장된 모델 파일 확인
    model_path = 'lorenz96_neural_dde_model.pth'
    if not os.path.exists(model_path):
        print(f"오류: 모델 파일 '{model_path}'을 찾을 수 없습니다.")
        print("먼저 학습을 완료하여 모델을 저장해주세요.")
        return
    
    # 모델 로딩
    print(f"[+] 모델 로딩 중...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 모델 초기화
    model = L96MZNetwork(K, F, delay_Tau, lag_M, hidden_dim, dt).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"모델 로딩 완료!")
    print(f"저장된 하이퍼파라미터:")
    for key, value in checkpoint['hyperparameters'].items():
        print(f"  {key}: {value}")
    
    # 스케일러 정보 확인
    if 'scaler_info' in checkpoint:
        print(f"\n[+] 스케일러 정보 로딩 완료:")
        scaler_info = checkpoint['scaler_info']
        print(f"  방법: {scaler_info['method']}")
        if scaler_info['method'] == 'standard':
            print(f"  평균: {scaler_info['mean']}")
            print(f"  표준편차: {scaler_info['scale']}")
        
        # 스케일러 재구성
        scaler = DataScaler(method=scaler_info['method'])
        if scaler_info['method'] == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler.scaler = StandardScaler()
            scaler.scaler.mean_ = scaler_info['mean']
            scaler.scaler.scale_ = scaler_info['scale']
            scaler.is_fitted = True
    else:
        print(f"\n[!] 경고: 스케일러 정보가 없습니다. 원본 데이터로 추론을 진행합니다.")
        scaler = None
    
    # 추론을 위한 테스트 데이터 생성
    print(f"\n[+] 추론용 테스트 데이터 생성 중...")
    
    # 간단한 테스트 데이터 생성 (Lorenz 96 시스템의 초기값)
    test_traj = np.load(os.path.join(os.getcwd(), "simulated_data", f"X_batch_coupled_1.npy"))[0]
    test_t = np.load(os.path.join(os.getcwd(), "simulated_data", f"t_batch_coupled_1.npy"))[0]
    
    if test_t[0] != 0:
        test_t = test_t - test_t[0]
    
    # 데이터 형태 변환
    if test_traj.ndim == 2:
        if test_traj.shape[0] == K:
            X_test = test_traj[:K, :].T
        elif test_traj.shape[1] == K:
            X_test = test_traj[:, :K]
        else:
            raise ValueError(f"test_traj shape={test_traj.shape}, cannot infer (T,K).")
    else:
        raise ValueError(f"test_traj ndim={test_traj.ndim}, expected 2D array.")
    
    print(f"테스트 데이터 형태: {X_test.shape}")
    print(f"원본 데이터 범위: [{np.min(X_test):.3f}, {np.max(X_test):.3f}]")
    
    # 스케일링 적용
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        print(f"스케일링된 데이터 범위: [{np.min(X_test_scaled):.3f}, {np.max(X_test_scaled):.3f}]")
    else:
        X_test_scaled = X_test
    
    # 추론 실행
    print(f"\n[+] 추론 실행 중...")
    
    with torch.no_grad():
        # 단일 trajectory에 대한 추론
        delays_vector = torch.tensor([(i + 1) * lag_M * dt for i in range(delay_Tau)], 
                                      dtype=torch.float32, device=device)
        
        # 초기 조건 설정 (스케일링된 데이터 사용)
        y0 = X_test_scaled[0:delay_Tau * lag_M + 1, :]  # 초기 히스토리
        history_fn = lambda t: torch.tensor(y0, dtype=torch.float32, device=device)
        
        # test_t를 torch tensor로 변환 (타입 명시)
        test_t_tensor = torch.tensor(test_t, dtype=torch.float32, device=device)
        
        # 전체 trajectory에 대해 추론하기 위해 더 긴 시간 구간 설정
        t_start = torch.tensor(test_t[0], dtype=torch.float32, device=device)
        t_end = torch.tensor(test_t[-1], dtype=torch.float32, device=device)
        
        print(f"추론 시간 구간: {t_start:.6f} ~ {t_end:.6f}")
        print(f"전체 시간점 수: {len(test_t)}")
        
        # 추론 실행 - 전체 시간 구간에 대해
        solution = integrate(
            func=model,
            solver=RK4(),
            t0=t_start,
            t1=t_end,
            ts=test_t_tensor,
            y0=history_fn,
            func_args=None,
            stepsize_controller=AdaptiveStepSizeController(rtol=1e-6, atol=1e-8),
            dt0=torch.tensor(dt, dtype=torch.float32, device=device),
            delays=delays_vector,
            discretize_then_optimize=False
        )
        
        # 결과를 CPU로 이동
        solution = solution.cpu().numpy()
        
        print(f"추론 완료! 결과 형태: {solution.shape}")
        print(f"test_t 형태: {test_t.shape}")
        print(f"X_test 형태: {X_test.shape}")
        
        # solution이 3차원인 경우 2차원으로 변환
        if solution.ndim == 3:
            print(f"solution을 2차원으로 변환: {solution.shape} -> {solution.shape[1], solution.shape[2]}")
            solution = solution[0]  # 첫 번째 배치만 사용
        
        # 차원 확인 및 수정
        if solution.shape[0] != test_t.shape[0]:
            print(f"경고: 시간 차원 불일치! solution: {solution.shape[0]}, test_t: {test_t.shape[0]}")
            # 시간 차원을 맞춤
            if solution.shape[0] < test_t.shape[0]:
                # solution이 더 짧은 경우, test_t를 잘라서 맞춤
                test_t_plot = test_t[:solution.shape[0]]
                X_test_plot = X_test[:solution.shape[0], :]
                print(f"시간 차원을 맞춤: test_t_plot: {test_t_plot.shape}, X_test_plot: {X_test_plot.shape}")
            else:
                # solution이 더 긴 경우, solution을 잘라서 맞춤
                solution = solution[:test_t.shape[0], :]
                print(f"solution을 잘라서 시간 차원 맞춤: {solution.shape}")
        else:
            test_t_plot = test_t
            X_test_plot = X_test
        
        # 스케일링 역변환 (원본 스케일로 복원)
        if scaler is not None:
            solution_original = scaler.inverse_transform(solution)
            print(f"역변환 완료! 원본 스케일로 복원되었습니다. 형태: {solution_original.shape}")
        else:
            solution_original = solution
        
        # 결과 시각화
        print(f"\n[+] 추론 결과 시각화...")
        
        # 차원 최종 확인
        print(f"최종 차원 확인:")
        print(f"  solution_original: {solution_original.shape}")
        print(f"  X_test_plot: {X_test_plot.shape}")
        print(f"  test_t_plot: {test_t_plot.shape}")
        
        # 차원이 일치하는지 확인
        if solution_original.shape != X_test_plot.shape:
            print(f"경고: 최종 차원 불일치! solution_original: {solution_original.shape}, X_test_plot: {X_test_plot.shape}")
            # 더 작은 차원에 맞춤
            min_time_steps = min(solution_original.shape[0], X_test_plot.shape[0])
            solution_original = solution_original[:min_time_steps, :]
            X_test_plot = X_test_plot[:min_time_steps, :]
            test_t_plot = test_t_plot[:min_time_steps]
            print(f"차원을 맞춤: {min_time_steps} 시간점으로 통일")
        
        plt.figure(figsize=(15, 12))
        
        # 첫 번째 변수에 대한 시계열 비교
        plt.subplot(3, 2, 1)
        plt.plot(test_t_plot, X_test_plot[:, 0], 'b-', label='real', linewidth=2)
        plt.plot(test_t_plot, solution_original[:, 0], 'r--', label='predicted', linewidth=2)
        plt.xlabel('time')
        plt.ylabel('X[0]')
        plt.title('first variable')
        plt.legend()
        plt.grid(True)
        
        # 두 번째 변수에 대한 시계열 비교
        plt.subplot(3, 2, 2)
        plt.plot(test_t_plot, X_test_plot[:, 1], 'b-', label='real', linewidth=2)
        plt.plot(test_t_plot, solution_original[:, 1], 'r--', label='predicted', linewidth=2)
        plt.xlabel('time')
        plt.ylabel('X[1]')
        plt.title('second variable')
        plt.legend()
        plt.grid(True)
        
        # 세 번째 변수에 대한 시계열 비교
        plt.subplot(3, 2, 3)
        plt.plot(test_t_plot, X_test_plot[:, 2], 'b-', label='real', linewidth=2)
        plt.plot(test_t_plot, solution_original[:, 2], 'r--', label='predicted', linewidth=2)
        plt.xlabel('time')
        plt.ylabel('X[2]')
        plt.title('third variable')
        plt.legend()
        plt.grid(True)
        
        # 네 번째 변수에 대한 시계열 비교
        plt.subplot(3, 2, 4)
        plt.plot(test_t_plot, X_test_plot[:, 3], 'b-', label='real', linewidth=2)
        plt.plot(test_t_plot, solution_original[:, 3], 'r--', label='predicted', linewidth=2)
        plt.xlabel('time')
        plt.ylabel('X[3]')
        plt.title('fourth variable')
        plt.legend()
        plt.grid(True)
        
        # 오차 분석
        plt.subplot(3, 2, 5)
        error = np.abs(solution_original - X_test_plot)
        plt.plot(test_t_plot, error[:, 0], 'g-', label='X[0] error', linewidth=2)
        plt.plot(test_t_plot, error[:, 1], 'm-', label='X[1] error', linewidth=2)
        plt.plot(test_t_plot, error[:, 2], 'c-', label='X[2] error', linewidth=2)
        plt.plot(test_t_plot, error[:, 3], 'y-', label='X[3] error', linewidth=2)
        plt.xlabel('time')
        plt.ylabel('absolute error')
        plt.title('error analysis')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        
        # 전체 변수에 대한 RMSE
        plt.subplot(3, 2, 6)
        rmse_per_var = np.sqrt(np.mean((solution_original - X_test_plot) ** 2, axis=0))
        plt.bar(range(K), rmse_per_var)
        plt.xlabel('variable index')
        plt.ylabel('RMSE')
        plt.title('RMSE per variable')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 통계 정보 출력
        print(f"\n=== 추론 결과 통계 ===")
        print(f"전체 RMSE: {np.sqrt(np.mean((solution_original - X_test_plot) ** 2)):.6f}")
        print(f"RMSE per variable:")
        rmse_per_var = np.sqrt(np.mean((solution_original - X_test_plot) ** 2, axis=0))
        for i in range(K):
            print(f"  X[{i}]: {rmse_per_var[i]:.6f}")
        
        print(f"maximum absolute error: {np.max(np.abs(solution_original - X_test_plot)):.6f}")
        print(f"mean absolute error: {np.mean(np.abs(solution_original - X_test_plot)):.6f}")
        
        # 시간별 오차 분석
        print(f"\n=== 시간별 오차 분석 ===")
        time_error = np.sqrt(np.mean((solution_original - X_test_plot) ** 2, axis=1))
        print(f"초기 시간 (t={test_t_plot[0]:.3f}): RMSE = {time_error[0]:.6f}")
        print(f"중간 시간 (t={test_t_plot[len(test_t_plot)//2]:.3f}): RMSE = {time_error[len(test_t_plot)//2]:.6f}")
        print(f"최종 시간 (t={test_t_plot[-1]:.3f}): RMSE = {time_error[-1]:.6f}")
        print(f"시간에 따른 RMSE 변화: {np.std(time_error):.6f}")
        
        # 스케일링 효과 분석
        if scaler is not None:
            print(f"\n=== scaling effect analysis ===")
            print(f"scaling method: {scaler_info['method']}")
            print(f"original data standard deviation: {np.std(X_test_plot):.6f}")
            print(f"scaled data standard deviation: {np.std(X_test_scaled[:solution.shape[0], :]):.6f}")
            print(f"scaling effect: data variance is adjusted to {np.std(X_test_scaled[:solution.shape[0], :])/np.std(X_test_plot):.3f} times")


if __name__ == "__main__":
    # 사용자 입력 받기
    print("=== Neural DDE for Lorenz 96 System ===")
    print("1. 학습 및 모델 저장")
    print("2. 모델 로딩 및 추론")
    
    while True:
        try:
            choice = input("\n선택하세요 (1 또는 2): ").strip()
            if choice in ['1', '2']:
                break
            else:
                print("1 또는 2를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            exit()
        except:
            print("잘못된 입력입니다. 1 또는 2를 입력해주세요.")
    
    # 스케일링 방법 선택 (학습 모드에서만)
    scaling_method = 'standard'
    if choice == '1':
        print(f"\n=== 스케일링 방법 선택 ===")
        print("1. Standard Scaler (Z-score 정규화)")
        print("2. MinMax Scaler (0-1 정규화)")
        print("3. Robust Scaler (이상치에 강건한 정규화)")
        
        while True:
            try:
                scaling_choice = input("\n스케일링 방법을 선택하세요 (1, 2, 또는 3): ").strip()
                if scaling_choice == '1':
                    scaling_method = 'standard'
                    print("Standard Scaler를 선택했습니다.")
                    break
                elif scaling_choice == '2':
                    scaling_method = 'minmax'
                    print("MinMax Scaler를 선택했습니다.")
                    break
                elif scaling_choice == '3':
                    scaling_method = 'robust'
                    print("Robust Scaler를 선택했습니다.")
                    break
                else:
                    print("1, 2, 또는 3를 입력해주세요.")
            except KeyboardInterrupt:
                print("\n프로그램을 종료합니다.")
                exit()
            except:
                print("잘못된 입력입니다. 1, 2, 또는 3를 입력해주세요.")
    
    # 하이퍼파라미터
    dt = 0.005
    batch_size = 64  # Neural DDE는 메모리 사용량이 많아서 작게 설정
    lr = 0.001
    max_epoch = 100
    delay_Tau = 2
    lag_M = 1
    K = 8  # X 변수 개수
    F = 15  # Lorenz 96 파라미터
    hidden_dim = 128

    print(f"\n=== 설정 정보 ===")
    print(f"delay_Tau: {delay_Tau} (과거 {delay_Tau}개 데이터 사용)")
    print(f"lag_M: {lag_M} (데이터 간격: {lag_M}*dt)")
    print(f"K: {K} (X 변수 개수)")
    print(f"F: {F} (강제 항)")
    print(f"Device: {device}")
    if choice == '1':
        print(f"스케일링 방법: {scaling_method}")

    if choice == '1':
        # 학습 및 모델 저장
        train_and_save_model(dt, batch_size, lr, max_epoch, delay_Tau, lag_M, K, F, hidden_dim, scaling_method)
    else:
        # 모델 로딩 및 추론
        load_and_inference_model(dt, delay_Tau, lag_M, K, F, hidden_dim)