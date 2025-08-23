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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class Lorenz96Dataset(Dataset):
    def __init__(self, time_axis, trajectory_value):
        self.ts = time_axis
        self.ys = trajectory_value

    def __getitem__(self, index):
        # ✅ 여기서 float32로 고정
        ts = torch.tensor(self.ts[index], dtype=torch.float32)
        ys = torch.tensor(self.ys[index], dtype=torch.float32)
        return ts, ys

    def __len__(self):
        return self.ys.shape[0]


def train_neural_dde(model, train_loader, optimizer, criterion, device, dt=0.005, delay_Tau=2, lag_M=2):
    """
    Neural DDE 모델 학습 (배치 단위 처리)
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    for batch_idx, (time_series, trajectory_batch) in enumerate(train_loader):
        delays_vector = torch.as_tensor([(i + 1) * lag_M *dt for i in range(delay_Tau)],
                                    dtype=time_series.dtype, device=device)
        # trajectory_batch: (B, time_steps, K) 형태
        B, time_steps, K = trajectory_batch.shape

        # 배치 단위로 데이터를 device로 이동
        trajectory_batch = trajectory_batch.to(device, dtype=time_series.dtype)  # (B, time_steps, K)

        optimizer.zero_grad()
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
            discretize_then_optimize=True)

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
            optimizer.zero_grad()
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
                discretize_then_optimize=True)
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



if __name__ == "__main__":
    # torchdde API 테스트    # 하이퍼파라미터
    dt=0.005
    batch_size = 64  # Neural DDE는 메모리 사용량이 많아서 작게 설정
    lr = 0.001
    max_epoch = 100
    delay_Tau = torch.nn.Parameter(torch.tensor(2))
    lag_M = 1
    K = 8  # X 변수 개수
    F = 15  # Lorenz 96 파라미터
    hidden_dim = 128

    print(f"=== Neural DDE for Lorenz 96 System ===")
    print(f"delay_Tau: {delay_Tau} (과거 {delay_Tau}개 데이터 사용)")
    print(f"lag_M: {lag_M} (데이터 간격: {lag_M}*dt)")
    print(f"K: {K} (X 변수 개수)")
    print(f"F: {F} (강제 항)")
    print(f"Device: {device}")

    # 데이터 생성
    print("\n[+] Trajectory 생성 중...")

    # 여러 개의 trajectory 생성 (여기서는 5개 생성)
    num_trajectories = 5
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

    # 데이터셋 생성
    print(f"\n[+] 데이터셋 생성 중...")
    dataset = Lorenz96Dataset(time_list, trajectories)
    print(f"데이터셋 크기: {len(dataset)}")

    # 학습/검증 분할 (trajectory 기반 분할)
    # trajectories.shape = (num_trajectories, time_steps, K)이므로
    # num_trajectories로 분할해야 함
    num_trajectories = trajectories.shape[0]
    train_size = int(0.8 * num_trajectories)
    val_size = num_trajectories - train_size

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, num_trajectories))

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # --- (2) DataLoader(선택적) 성능 옵션 ---
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == "cuda"), num_workers=2, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        pin_memory=(device.type == "cuda"), num_workers=2, drop_last=False
    )
    print(f"학습 데이터: {len(train_dataset)}, 검증 데이터: {len(val_dataset)}")

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
        val_loss = evaluate_neural_dde(model, val_loader, criterion, device, dt, delay_Tau, lag_M)

        # 학습률 조정
        scheduler.step(val_loss)

        # 결과 저장
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{max_epoch}: "
              f"Train Loss = {train_loss:.6f}, "
              f"Val Loss = {val_loss:.6f}, "
              f"Time = {epoch_time:.2f}s")


        # 조기 종료 체크
        if val_loss < 1e-6:
            print("검증 손실이 충분히 작아졌습니다. 학습을 종료합니다.")
            break

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

    # 모델 저장
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
        }
    }, 'lorenz96_neural_dde_model.pth')

    print(f"모델이 'lorenz96_neural_dde_model.pth'에 저장되었습니다.")