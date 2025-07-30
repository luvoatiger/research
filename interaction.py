import os
import numpy as np
import torch
import torch.nn as nn
from torchdde import integrate, AdaptiveStepSizeController, Dopri5
from tqdm import tqdm
import json
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===========================
# 1. NDDE 모델 정의 (수학적 정의에 따른 구현)
# ===========================
class NDDE(nn.Module):
    """
    Neural Delay Differential Equation (NDDE) 모델
    수학적 정의: dy/dt = f_theta(t, y(t), y(t-tau_1), ..., y(t-tau_n))

    Args:
        delays: 지연 시간들의 리스트 [tau_1, tau_2, ..., tau_n]
        in_size: y(t)의 차원
        out_size: dy/dt의 차원 (보통 in_size와 같음)
        width_size: 은닉층의 너비
        depth: 은닉층의 깊이
    """
    def __init__(self, delays, in_size, out_size, width_size=128, depth=3):
        super().__init__()
        self.delays = delays
        self.in_size = in_size
        self.out_size = out_size

        # 입력 차원: 현재 상태 + 모든 지연 상태들
        # in_dim = in_size * (1 + len(delays))
        self.in_dim = in_size * (1 + len(delays))

        # MLP 구성: depth개의 은닉층 + 출력층
        layers = []
        # 첫 번째 은닉층
        layers.append(nn.Linear(self.in_dim, width_size))
        layers.append(nn.LeakyReLU())

        # 중간 은닉층들
        for _ in range(depth - 1):
            layers.append(nn.Linear(width_size, width_size))
            layers.append(nn.LeakyReLU())

        # 출력층
        layers.append(nn.Linear(width_size, out_size))

        self.mlp = nn.Sequential(*layers)

        # 가중치 초기화
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """가중치 초기화"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, t, z, func_args, *, history):
        """
        NDDE forward pass

        Args:
            t: 현재 시간
            z: 현재 상태 y(t) [batch_size, in_size]
            func_args: 추가 인수 (사용하지 않음)
            history: 지연 상태들의 리스트 [y(t-tau_1), y(t-tau_2), ..., y(t-tau_n)]
                    각각 [batch_size, in_size] 형태

        Returns:
            dy/dt: [batch_size, out_size]
        """
        # 입력 데이터 검증
        if torch.isnan(z).any() or torch.isinf(z).any():
            print(f"경고: NDDE 현재 상태에 NaN/Inf가 포함되어 있습니다!")
            return torch.zeros(z.shape[0], self.out_size, device=z.device)

        for i, hist in enumerate(history):
            if torch.isnan(hist).any() or torch.isinf(hist).any():
                print(f"경고: NDDE 지연 상태 {i}에 NaN/Inf가 포함되어 있습니다!")
                return torch.zeros(z.shape[0], self.out_size, device=z.device)

        # 현재 상태와 모든 지연 상태들을 연결
        # torch.cat([z, *history], dim=-1) 형태로 구현
        concatenated = torch.cat([z, *history], dim=-1)

        # MLP를 통한 f_theta 계산
        output = self.mlp(concatenated)

        # 출력 데이터 검증
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"경고: NDDE 출력에 NaN/Inf가 포함되어 있습니다!")
            return torch.zeros(z.shape[0], self.out_size, device=z.device)

        return output


# ===========================
# 2. Lorenz 96 시스템과 NDDE 결합
# ===========================
class Lorenz96NDDE(nn.Module):
    """
    Lorenz 96 시스템과 NDDE를 결합한 모델
    dx/dt = Lorenz96_markov(x) + NDDE_memory(x, x(t-tau_1), ..., x(t-tau_n))
    """
    def __init__(self, delays, K, J, F, h, b, c, width_size=128, depth=3):
        super().__init__()
        self.K = K
        self.J = J
        self.delays = delays

        # Lorenz 96 시스템 파라미터 (metadata에서 가져옴)
        self.F = F      # 강제 항 파라미터
        self.h = h      # 결합 강도
        self.b = b      # Y 변수 스케일링
        self.c = c      # Y 변수 시간 스케일

        # NDDE 메모리 항
        self.ndde = NDDE(delays=delays, in_size=K, out_size=K,
                        width_size=width_size, depth=depth)

    def forward(self, t, x, func_args, *, history):
        """
        Lorenz 96 + NDDE forward pass

        Args:
            t: 현재 시간
            x: 현재 상태 [batch_size, K]
            func_args: 추가 인수
            history: 지연 상태들의 리스트

        Returns:
            dx/dt: [batch_size, K]
        """
        # Lorenz 96 Markov 항 계산
        markov_term = self.lorenz96_markov(x)

        # NDDE 메모리 항 계산
        memory_term = self.ndde(t, x, func_args, history=history)

        # 전체 미분 계산
        dxdt = markov_term + memory_term

        return dxdt

    def lorenz96_markov(self, x):
        """
        Lorenz 96 시스템의 Markov 항 계산
        metadata의 파라미터들을 사용하여 정확한 Lorenz 96 시스템 구현
        """
        K = x.shape[1]
        roll_p1 = torch.roll(x, shifts=-1, dims=1)
        roll_m2 = torch.roll(x, shifts=2, dims=1)
        roll_m1 = torch.roll(x, shifts=1, dims=1)

        # 정확한 Lorenz 96 Markov 항 계산
        # dX[k] = (X[(k+1) % K] - X[(k-2) % K]) * X[(k-1) % K] - X[k] + F
        result = (roll_p1 - roll_m2) * roll_m1 - x + self.F

        return result


# ===========================
# 3. 데이터 전처리
# ===========================
def load_l96_data(data_dir: str, N: int = 300):
    """
    Lorenz 96 데이터 로딩

    Args:
        data_dir (str): 데이터 디렉토리 경로
        N (int): 로딩할 배치 수

    Returns:
        list: 데이터 리스트
    """
    data_list = []
    print(f"[+] 데이터 로딩 시작: {data_dir}")

    for i in range(1, N + 1):
        x_data_path = os.path.join(data_dir, f"X_batch_{i}.npy")
        t_data_path = os.path.join(data_dir, f"t_batch_{i}.npy")
        if not os.path.exists(x_data_path):
            print(f"경고: 파일이 존재하지 않습니다: {x_data_path}")
            continue

        try:
            X = np.load(x_data_path)[0]  # shape: [T, K]
            t = np.load(t_data_path)[0]  # shape: [T]

            # 데이터 검증
            if np.isnan(X).any() or np.isinf(X).any():
                print(f"경고: 배치 {i}에 NaN/Inf가 포함되어 있습니다!")
                print(f"데이터 범위: [{X.min():.4f}, {X.max():.4f}]")
                continue

            data_list.append((X, t) )

        except Exception as e:
            print(f"경고: 배치 {i} 로딩 중 오류 발생: {e}")
            continue

    print(f"[+] 성공적으로 로딩된 배치 수: {len(data_list)}/{N}")
    if len(data_list) == 0:
        raise ValueError("로딩된 데이터가 없습니다!")

    return data_list


def preprocess_trajectories_for_ndde(data_list, sequence_length=200, num_sequences_per_traj=3, dt=0.01, seed=42, use_real_time=False):
    """
    로딩된 trajectory들을 Neural DDE 학습에 맞게 전처리
    각 trajectory에서 랜덤하게 슬라이싱하여 sequence들을 생성

    Args:
        data_list: load_l96_data로 로딩된 trajectory 리스트 (X, t) 튜플
        sequence_length: 각 sequence의 길이
        num_sequences_per_traj: 각 trajectory에서 추출할 sequence 개수
        dt: 시간 간격 (use_real_time=False일 때만 사용)
        seed: 랜덤 시드
        use_real_time: 실제 시간 데이터 사용 여부

    Returns:
        tuple: (ts, ys) - 시간 구간과 전처리된 sequence들
    """
    np.random.seed(seed)
    print(f"[+] Neural DDE용 데이터 전처리 시작...")
    print(f"  - 원본 trajectory 수: {len(data_list)}")
    print(f"  - Sequence 길이: {sequence_length}")
    print(f"  - Trajectory당 sequence 수: {num_sequences_per_traj}")
    print(f"  - 실제 시간 사용: {use_real_time}")

    processed_sequences = []
    processed_times = []  # 실제 시간 데이터 저장
    total_sequences = 0

    for i, (X, t) in enumerate(data_list):
        # 데이터 검증
        if np.isnan(X).any() or np.isinf(X).any():
            print(f"경고: trajectory {i}에 NaN/Inf가 포함되어 있습니다!")
            continue

        T = len(X)
        
        # sequence_length보다 짧은 trajectory는 건너뛰기
        if T < sequence_length:
            print(f"경고: trajectory {i}가 너무 짧습니다 (길이: {T})")
            continue

        # 겹치지 않는 랜덤 시작점들 생성
        max_start_idx = T - sequence_length
        if max_start_idx < num_sequences_per_traj:
            # 가능한 sequence 수가 요청된 수보다 적은 경우
            num_sequences = max_start_idx + 1
            start_indices = list(range(num_sequences))
        else:
            # 랜덤하게 시작점 선택 (겹치지 않도록)
            start_indices = np.random.choice(max_start_idx + 1, 
                                           size=min(num_sequences_per_traj, max_start_idx + 1), 
                                           replace=False)

        # 각 시작점에서 sequence 추출
        for start_idx in start_indices:
            end_idx = start_idx + sequence_length
            
            # sequence 추출
            sequence = X[start_idx:end_idx]
            
            # 데이터 검증
            if np.isnan(sequence).any() or np.isinf(sequence).any():
                print(f"경고: trajectory {i}의 sequence {start_idx}-{end_idx}에 NaN/Inf가 포함되어 있습니다!")
                continue
            
            processed_sequences.append(sequence)
            
            # 실제 시간 데이터도 추출
            if use_real_time:
                time_sequence = t[start_idx:end_idx]
                processed_times.append(time_sequence)
            
            total_sequences += 1

    if len(processed_sequences) == 0:
        raise ValueError("전처리된 유효한 sequence가 없습니다!")

    # 시간 구간 생성
    if use_real_time:
        # 실제 시간 데이터 사용 (첫 번째 sequence의 시간을 기준으로)
        ts = torch.tensor(processed_times[0], dtype=torch.float32)
        print(f"  - 실제 시간 범위: [{ts[0]:.4f}, {ts[-1]:.4f}]")
    else:
        # 균등한 시간 간격 생성
        ts = torch.linspace(0, (sequence_length-1) * dt, sequence_length)
        print(f"  - 균등 시간 간격: [{ts[0]:.4f}, {ts[-1]:.4f}]")

    # 배치로 스택
    ys = torch.tensor(np.stack(processed_sequences), dtype=torch.float32)  # [N, T, K]

    print(f"[+] 데이터 전처리 완료:")
    print(f"  - 생성된 sequence 수: {ys.shape[0]}")
    print(f"  - 시간 스텝: {ys.shape[1]}")
    print(f"  - 변수 수: {ys.shape[2]}")
    print(f"  - 데이터 범위: [{ys.min():.4f}, {ys.max():.4f}]")
    print(f"  - 총 sequence 수: {total_sequences}")

    return ts, ys


def create_fixed_delays(dt=0.01, num_delays=5):
    """
    고정된 지연 시간 생성 (dt에 맞게 설정)

    Args:
        dt: 시간 간격
        num_delays: 지연 시간 개수

    Returns:
        torch.tensor: 고정된 지연 시간들
    """
    # dt에 맞게 1, 2, 3, 4, 5 스텝 뒤의 과거 값들을 가져오도록 설정
    delays = torch.tensor([np.round(dt * i, 4) for i in range(1, num_delays + 1)], dtype=torch.float32)

    print(f"[+] 고정 지연 시간 생성:")
    print(f"  - 지연 시간 개수: {num_delays}")
    print(f"  - 시간 간격 (dt): {dt}")
    print(f"  - 지연 시간들: {delays.tolist()}")

    return delays


class Lorenz96Dataset(torch.utils.data.Dataset):
    """
    Lorenz 96 데이터셋
    """
    def __init__(self, ys):
        self.ys = ys

    def __getitem__(self, index):
        return self.ys[index]

    def __len__(self):
        return self.ys.shape[0]


# ===========================
# 5. NDDE 학습 함수 (torchdde 기반)
# ===========================
def train_lorenz96_ndde(
    metadata,
    dataset,
    delays,
    batch_size=32,
    lr=0.001,
    max_epoch=50,
    width_size=128,
    depth=3,
    seed=42,
    plot=True,
    print_every=5,
    device="cpu"
):
    """
    Lorenz96NDDE 모델 학습 함수 (torchdde 기반)

    Args:
        metadata: Lorenz 96 시스템 메타데이터
        dataset: 전처리된 데이터셋
        delays: 학습 가능한 지연 시간 파라미터
        batch_size: 배치 크기
        lr: 학습률
        max_epoch: 최대 에포크 수
        width_size: 은닉층 너비
        depth: 은닉층 깊이
        seed: 랜덤 시드
        plot: 시각화 여부
        print_every: 로깅 간격
        device: 디바이스

    Returns:
        tuple: (ts, ys, model, losses, delays_evol)
    """
    torch.manual_seed(seed)

    # 데이터 추출
    ts = dataset.ts
    ys = dataset.ys

    # 모델 생성
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']

    model = Lorenz96NDDE(
        delays=delays,
        K=K, J=J, F=F, h=h, b=b, c=c,
        width_size=width_size, depth=depth
    ).to(device)

    print(f"[+] 모델 구성 완료:")
    print(f"  - 모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")

    # 데이터로더 생성
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 학습 설정
    losses, delays_evol = [], []
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    print(f"[+] 학습 시작: {max_epoch} 에포크, 배치 크기 {batch_size}")

    # 학습 루프
    for epoch in tqdm(range(max_epoch)):
        model.train()
        epoch_losses = []

        for step, data in enumerate(train_loader):
            start_time = time.time()
            optimizer.zero_grad()

            data = data.to(device)  # [batch_size, T, K]

            # History 함수: 각 배치의 첫 번째 시간 스텝을 초기 조건으로 사용
            history_fn = lambda t: data[:, 0]  # [batch_size, K]

            try:
                # NDDE 통합
                ys_pred = integrate(
                    model,
                    Dopri5(),
                    ts[0],
                    ts[-1],
                    ts,
                    history_fn,
                    func_args=None,
                    dt0=ts[1] - ts[0],
                    stepsize_controller=AdaptiveStepSizeController(1e-6, 1e-9),
                    discretize_then_optimize=True,
                    delays=delays,
                )

                # Loss 계산
                loss = loss_fn(ys_pred, data)
                epoch_losses.append(loss.item())

                # 역전파
                loss.backward()

                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # 지연 시간은 고정이므로 기록하지 않음
                # delays_evol.append(delays.clone().detach())  # 주석 처리

            except Exception as e:
                print(f"경고: 배치 {step} 처리 중 오류 발생: {e}")
                continue

        # 에포크 평균 손실 계산
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            scheduler.step(avg_loss)

            # 로깅
            if (epoch % print_every) == 0 or epoch == max_epoch - 1:
                print(
                    f"Epoch: {epoch:3d}/{max_epoch}, "
                    f"Loss: {avg_loss:.6f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

    print(f"[+] 학습 완료! 최종 손실: {losses[-1]:.6f}")

    # 시각화
    if plot:
        plot_training_results(ts, ys, model, losses, [], device)  # delays_evol은 빈 리스트로 전달

    return ts, ys, model, losses, []


def plot_training_results(ts, ys, model, losses, delays_evol, device):
    """
    학습 결과 시각화
    """
    model.eval()

    # 테스트 예측
    with torch.no_grad():
        test_data = ys[0:1].to(device)  # 첫 번째 trajectory
        history_fn = lambda t: test_data[:, 0]

        ys_pred = integrate(
            model,
            Dopri5(),
            ts[0],
            ts[-1],
            ts,
            history_fn,
            func_args=None,
            dt0=ts[1] - ts[0],
            stepsize_controller=AdaptiveStepSizeController(1e-6, 1e-9),
            delays=model.delays,
        )

    # 변수 수 계산
    num_variables = ys.shape[2]
    
    # 시각화 1: 모든 변수들의 개별 subplot
    fig1, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i in range(num_variables):
        ax = axes[i]
        ax.plot(ts.cpu(), ys_pred[0, :, i].cpu(), '--', c='red', label='NDDE Prediction', linewidth=2)
        ax.plot(ts.cpu(), test_data[0, :, i].cpu(), '-', c='blue', label='Ground Truth', linewidth=2)
        ax.set_xlabel('Time t')
        ax.set_ylabel(f'X_{i+1}(t)')
        ax.set_title(f'Variable X_{i+1}: Prediction vs Ground Truth')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('lorenz96_ndde_all_variables.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 시각화 2: 학습 결과 요약
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 손실 곡선
    axes[0, 0].plot(losses)
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)

    # 2. 예측 오차
    error = (ys_pred[0] - test_data[0]).abs().mean(dim=1)
    axes[0, 1].plot(ts.cpu(), error.cpu())
    axes[0, 1].set_xlabel('Time t')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    axes[0, 1].set_title('Prediction Error over Time')
    axes[0, 1].grid(True)

    # 3. RMSE by Variable
    mse = torch.mean((ys_pred[0] - test_data[0]) ** 2, dim=0)
    rmse = torch.sqrt(mse)
    axes[1, 0].bar(range(1, len(rmse) + 1), rmse.cpu())
    axes[1, 0].set_xlabel('Variable Index')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].set_title('RMSE by Variable')
    axes[1, 0].grid(True)

    # 4. 모든 변수 비교 (하나의 그래프에)
    for i in range(num_variables):
        axes[1, 1].plot(ts.cpu(), ys_pred[0, :, i].cpu(), '--', linewidth=1, alpha=0.7, label=f'Pred X_{i+1}')
        axes[1, 1].plot(ts.cpu(), test_data[0, :, i].cpu(), '-', linewidth=1, alpha=0.7, label=f'True X_{i+1}')
    axes[1, 1].set_xlabel('Time t')
    axes[1, 1].set_ylabel('X(t)')
    axes[1, 1].set_title('All Variables Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('lorenz96_ndde_training_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


    # 성능 지표 출력
    print(f"\n[+] 최종 성능 지표:")
    print(f"  - 전체 RMSE: {torch.sqrt(torch.mean((ys_pred[0] - test_data[0]) ** 2)):.6f}")

    for i in range(len(rmse)):
        print(f"  - 변수 {i+1} RMSE: {rmse[i]:.6f}")


# ===========================
# 6. 전체 실행
# ===========================
if __name__ == "__main__":
    # -------------------------------
    # ⚙️ 설정
    # -------------------------------
    data_dir = os.path.join(os.getcwd(), "simulated_data")  # 데이터 경로
    hidden_dim = 128
    epochs = 100
    device = "cpu"
    dropout_rate = 0.1
    sequence_length = 200  # 각 sequence의 길이
    num_sequences_per_traj = 3  # 각 trajectory에서 추출할 sequence 개수
    use_real_time = False  # 실제 시간 데이터 사용 여부 (True: 실제 시간, False: 균등 간격)

    # -------------------------------
    # 📂 데이터 로딩
    # -------------------------------
    data_list = load_l96_data(data_dir)

    # -------------------------------
    # 📂 메타데이터 로딩
    # -------------------------------
    try:
        # 메타데이터 로딩
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)

        # 시스템 파라미터 추출
        K = metadata['K']
        J = metadata['J']
        F = metadata['F']
        h = metadata['h']
        b = metadata['b']
        c = metadata['c']
        dt = metadata['dt']

        print(f"[+] 메타데이터 로딩 완료:")
        print(f"  - K (X 변수 수): {K}")
        print(f"  - J (Y 변수 수): {J}")
        print(f"  - F (강제 항): {F}")
        print(f"  - h (결합 강도): {h}")
        print(f"  - b (Y 스케일링): {b}")
        print(f"  - c (Y 시간 스케일): {c}")
        print(f"  - dt (시간 간격): {dt}")

    except Exception as e:
        print(f"오류: 메타데이터 로딩 실패: {e}")
        exit(1)

    # -------------------------------
    # 🔄 데이터 전처리
    # -------------------------------
    try:
        print(f"[+] 데이터 전처리 시작...")

        # Neural DDE용 데이터 전처리
        ts, ys = preprocess_trajectories_for_ndde(
            data_list,
            sequence_length=sequence_length,
            num_sequences_per_traj=num_sequences_per_traj,
            dt=dt,
            seed=42,
            use_real_time=use_real_time
        )

        # 데이터셋 생성
        dataset = Lorenz96Dataset(ys)
        dataset.ts = ts  # 시간 구간을 데이터셋에 추가

        print(f"[+] 데이터셋 생성 완료:")
        print(f"  - 데이터셋 크기: {len(dataset)}")
        print(f"  - 데이터 형태: {ys.shape}")

    except Exception as e:
        print(f"오류: 데이터 전처리 실패: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # -------------------------------
    # ⏰ 지연 시간 파라미터 생성
    # -------------------------------
    try:
        # 고정된 지연 시간 생성
        delays = create_fixed_delays(
            dt=dt,
            num_delays=5
        )

    except Exception as e:
        print(f"오류: 지연 시간 생성 실패: {e}")
        exit(1)

    # -------------------------------
    # 🧠 NDDE 모델 학습
    # -------------------------------
    try:
        print(f"[+] 설정:")
        print(f"  - 데이터 디렉토리: {data_dir}")
        print(f"  - 은닉층 차원: {hidden_dim}")
        print(f"  - 학습 에포크: {epochs}")
        print(f"  - 디바이스: {device}")
        print(f"  - Dropout 비율: {dropout_rate}")
        print(f"  - Sequence 길이: {sequence_length}")
        print(f"  - Trajectory당 sequence 수: {num_sequences_per_traj}")
        print(f"  - 실제 시간 사용: {use_real_time}")

        # Lorenz96NDDE 학습
        ts, ys, model, losses, delays_evol = train_lorenz96_ndde(
            metadata=metadata,
            dataset=dataset,
            delays=delays,
            batch_size=512,    # 작은 배치 크기
            lr=0.001,
            max_epoch=epochs,
            width_size=hidden_dim,
            depth=3,
            seed=42,
            plot=True,
            print_every=2,
            device=device
        )

        print(f"[+] NDDE 학습 완료!")
        print(f"  - 최종 손실: {losses[-1]:.6f}")
        print(f"  - 고정 지연 시간: {delays.tolist()}")

        # 모델 저장
        torch.save({
            'model_state_dict': model.state_dict(),
            'metadata': metadata,
            'losses': losses,
            'delays_evol': delays_evol,
            'final_delays': delays.tolist()
        }, 'lorenz96_ndde_model.pth')
        print(f"[+] 모델이 lorenz96_ndde_model.pth에 저장되었습니다.")

    except Exception as e:
        print(f"오류: NDDE 학습 실패: {e}")
        import traceback
        traceback.print_exc()
        exit(1)