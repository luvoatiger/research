"""
Baseline 모델(Learning subgrid-scale models with neural ordinary differential equations, Kim et al., 2023) 구현
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch


class SubgridNN(torch.nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128, output_dim=36):
        """
        Lorenz 96 시스템의 커플링 항을 근사하는 신경망
        
        Args:
            input_dim (int): 입력 차원 (X 변수의 차원)
            hidden_dim (int): 은닉층의 뉴런 수
            output_dim (int): 출력 차원 (커플링 항의 차원)
        """
        super(SubgridNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = torch.nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        """
        신경망 순전파
        
        Args:
            x (torch.Tensor): 입력 텐서, 형태: [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 예측된 커플링 항, 형태: [batch_size, output_dim]
        """
        x = self.fc1(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        
        # 커플링 항(h*Ysummed/b) 반환
        return x


class NeuralLorenz96(torch.nn.Module):
    def __init__(self, subgrid_nn):
        super(NeuralLorenz96, self).__init__()
        self.subgrid_nn = subgrid_nn


    def neural_L96_2t_xdot_ydot(self, X, Y, K, J, h, F, c, b, neural_coupling):
        """
        Lorenz-96 two-timescale 모델의 미분 방정식을 계산하는 함수 (신경망 서브그리드 모델 사용)
        
        Args:
            X: 큰 스케일 변수 (shape: [batch_size, length, K])
            Y: 작은 스케일 변수 (shape: [batch_size, length, K*J])
            K: 큰 스케일 변수의 수
            J: 각 큰 스케일 변수에 연결된 작은 스케일 변수의 수
            h: 커플링 계수
            F: 외부 강제력
            c: 시간 스케일 비율
            b: 공간 스케일 비율
            neural_coupling: 신경망 커플링 항

        Returns:
            Xdot: X의 시간 미분 (shape: [batch_size, length, K])
            Ydot: Y의 시간 미분 (shape: [batch_size, length, K*J])
        """
        # 모든 입력이 PyTorch 텐서인지 확인
        device = X.device
        if not isinstance(F, torch.Tensor):
            F = torch.tensor(F, dtype=torch.float32, device=device)
        if not isinstance(h, torch.Tensor):
            h = torch.tensor(h, dtype=torch.float32, device=device)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float32, device=device)
        
        
        # X의 미분 계산 (PyTorch 연산 사용)
        X_minus_1 = torch.roll(X, shifts=1, dims=0)
        X_plus_1 = torch.roll(X, shifts=-1, dims=0)
        X_minus_2 = torch.roll(X, shifts=2, dims=0)

        Xdot = X_minus_1 * (X_plus_1 - X_minus_2) - X + F + neural_coupling
        
        # Y의 미분 계산
        # Y_plus_1, Y_plus_2, Y_minus_1 계산 (마지막 차원에 대해 roll)
        Y_plus_1 = torch.roll(Y, shifts=-1, dims=-1)
        Y_plus_2 = torch.roll(Y, shifts=-2, dims=-1)
        Y_minus_1 = torch.roll(Y, shifts=1, dims=-1)
        
        # X를 반복하여 Y와 같은 차원으로 확장
        X_repeated = torch.repeat_interleave(X, J, dim=-1)
          
        # Ydot 계산
        hcJ = h * c / J
        Ydot = -c * J * Y_plus_1 * (Y_plus_2 - Y_minus_1) - c * Y + hcJ * X_repeated
        
        return Xdot, Ydot, neural_coupling


    def integrate_L96_2t_with_neural_coupling(self, X0, Y0, F, h, b, c, dt=0.001, neural_coupling=None):
        """
        Integrates forward-in-time the two time-scale Lorenz 1996 model, using the RK4 integration method.
        Returns the full history with nt+1 values starting with initial conditions, X[:,0]=X0 and Y[:,0]=Y0,
        and ending with the final state, X[:,nt+1] and Y[:,nt+1] at time t0+nt*si.

        Note the model is intergratedL

        Args:
            X0 : Values of X variables at the current time
            Y0 : Values of Y variables at the current time
            F  : Forcing term
            h  : coupling coefficient
            b  : ratio of amplitudes
            c  : time-scale ratio
            dt : The actual time step. If dt<si, then si is used. Otherwise si/dt must be a whole number. Default 0.001.

        Returns:
            X[:,:], Y[:,:], time[:], hcbY[:,:] : the full history X[n,k] and Y[n,k] at times t[n], and coupling term

        Example usage:
            X,Y,t,_ = integrate_L96_2t_with_coupling(5+5*np.random.rand(8), np.random.rand(8*4), 0.01, 500, 18, 1, 10, 10)
            plt.plot( t, X);
        """

        xhist, yhist = torch.zeros(X0.shape), torch.zeros(Y0.shape)

        X, Y = X0[0], Y0[0]
        xhist[0] = X
        yhist[0] = Y

        K = X0.shape[1]
        J = Y0.shape[1] // X0.shape[1]

        for n in range(1, X0.shape[0]):
            # RK4 update of X,Y
            Xdot1, Ydot1, _ = self.neural_L96_2t_xdot_ydot(X, Y, K, J, h, F, c, b, neural_coupling[n-1])
            Xdot2, Ydot2, _ = self.neural_L96_2t_xdot_ydot(
                X + 0.5 * dt * Xdot1, Y + 0.5 * dt * Ydot1, K, J, h, F, c, b, neural_coupling[n-1]
            )
            Xdot3, Ydot3, _ = self.neural_L96_2t_xdot_ydot(
                X + 0.5 * dt * Xdot2, Y + 0.5 * dt * Ydot2, K, J, h, F, c, b, neural_coupling[n-1]
            )
            Xdot4, Ydot4, _ = self.neural_L96_2t_xdot_ydot(
                X + dt * Xdot3, Y + dt * Ydot3, K, J, h, F, c, b, neural_coupling[n-1]
            )
            X = X + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
            Y = Y + (dt / 6.0) * ((Ydot1 + Ydot4) + 2.0 * (Ydot2 + Ydot3))

            xhist[n], yhist[n] = (X, Y)

        return xhist, yhist


def calculate_loss(batch_X_data, batch_Y_data, predicted_X_data, predicted_Y_data):
    batch_size = len(predicted_X_data)
    m_delta_t = len(predicted_X_data[0])

    predicted_X_tensor = torch.stack(predicted_X_data)
    predicted_Y_tensor = torch.stack(predicted_Y_data)

    # X와 Y를 합쳐서 Z 텐서 생성
    predicted_Z_tensor = torch.cat([predicted_X_tensor, predicted_Y_tensor], dim=-1)
    batch_Z_data = torch.cat([batch_X_data, batch_Y_data], dim=-1)

    # Z에 대한 제곱 오차 계산
    Z_squared_norms = torch.sum((predicted_Z_tensor - batch_Z_data)**2, dim=-1)  # [batch_size, m_delta_t]

    # MSE 손실 계산: (1/nm) * sum_{i=1}^n sum_{t=1}^m ||Z_t^(i) - Z_hat_t^(i)||^2
    loss = torch.sum(Z_squared_norms) / (batch_size * m_delta_t)

    return loss


def compare_coupling_terms(model_path, data_dir, batch_indices=None, time_steps=1000, save_path=None):
    """
    학습 데이터를 이용하여 학습된 커플링 항과 실제 커플링 항을 비교하는 함수
    
    Args:
        model_path: 학습된 모델 경로
        data_dir: 학습 데이터가 저장된 디렉토리
        batch_indices: 비교할 배치 인덱스 리스트 (None인 경우 첫 번째 배치 사용)
        time_steps: 시각화할 시간 단계 수
        save_path: 그래프 저장 경로 (None인 경우 저장하지 않음)
    
    Returns:
        fig1, fig2: 생성된 그래프 객체들
    """
    # 모델 로드
    subgrid_nn = SubgridNN()
    model = NeuralLorenz96(subgrid_nn)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 평가 모드로 설정
    
    # 배치 인덱스가 지정되지 않은 경우 첫 번째 배치 사용
    if batch_indices is None:
        batch_indices = [1]  # 첫 번째 배치
    
    # 데이터 로드 및 준비
    all_X_data = []
    all_true_coupling = []
    
    for idx in batch_indices:
        X_data = np.load(os.path.join(data_dir, f"X_batch_{idx}.npy"))
        C_data = np.load(os.path.join(data_dir, f"C_batch_{idx}.npy"))
        
        all_X_data.append(X_data)
        all_true_coupling.append(C_data)
    
    # 데이터 결합
    X_data = np.concatenate(all_X_data, axis=0)
    true_coupling = np.concatenate(all_true_coupling, axis=0)
    
    # 텐서로 변환
    X_tensor = torch.from_numpy(X_data).float()
    true_coupling_tensor = torch.from_numpy(true_coupling).float()
    
    # 배치 차원 제거 (필요한 경우)
    if X_tensor.dim() > 2:
        X_tensor = X_tensor.reshape(-1, X_tensor.shape[-1])
        true_coupling_tensor = true_coupling_tensor.reshape(-1, true_coupling_tensor.shape[-1])
    
    # 예측된 커플링 항 계산
    with torch.no_grad():
        predicted_coupling_tensor = model.subgrid_nn(X_tensor)
    
    # NumPy 배열로 변환
    true_coupling_np = true_coupling_tensor.numpy()
    predicted_coupling_np = predicted_coupling_tensor.numpy()
    
    # 차이 계산
    difference = true_coupling_np - predicted_coupling_np
    
    
    # 시각화 1: 커플링 항 비교
    fig1, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 컬러맵 설정
    cmap1 = plt.cm.viridis
    cmap2 = plt.cm.coolwarm
    
    # 실제 커플링 항 (S)
    im1 = axes[0].imshow(true_coupling_np.T, aspect='auto', cmap=cmap1, 
                         interpolation='none', origin='lower')
    axes[0].set_ylabel('$S$')
    axes[0].set_title('True Coupling Term')
    plt.colorbar(im1, ax=axes[0])
    
    # 예측된 커플링 항 (S_θ)
    im2 = axes[1].imshow(predicted_coupling_np.T, aspect='auto', cmap=cmap1, 
                         interpolation='none', origin='lower')
    axes[1].set_ylabel('$S_\\theta$')
    axes[1].set_title('Predicted Coupling Term (Neural Network)')
    plt.colorbar(im2, ax=axes[1])
    
    # 차이 (S - S_θ)
    # 차이의 최대 절대값을 기준으로 컬러맵 범위 설정
    max_diff = max(abs(difference.min()), abs(difference.max()))
    im3 = axes[2].imshow(difference.T, aspect='auto', cmap=cmap2, 
                         interpolation='none', origin='lower',
                         vmin=-max_diff, vmax=max_diff)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('$S - S_\\theta$')
    axes[2].set_title(f'Difference')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 시각화 2: 차이값의 분포도
    fig2, axes2 = plt.subplots(figsize=(10, 10))
    
    # 차이값 히스토그램 및 KDE
    sns.histplot(difference.flatten(), kde=True, ax=axes2, color='blue', bins=100)
    axes2.set_title('Distribution of Differences (S - S_θ)')
    axes2.set_xlabel('Difference Value')
    axes2.set_ylabel('Frequency')
       
    plt.tight_layout()
    
    # 분포도 그래프 저장
    if save_path:
        distribution_save_path = save_path.replace('.png', '_distribution.png')
        plt.savefig(distribution_save_path, dpi=300, bbox_inches='tight')
    
#    plt.show()
    
    print(f"차이값 통계:")
    print(f"  최소값: {difference.min():.4f}")
    print(f"  최대값: {difference.max():.4f}")
    print(f"  평균: {difference.mean():.4f}")
    print(f"  표준편차: {difference.std():.4f}")
        
    return fig1, fig2
    

def plot_hovmoller_diagram(model_path, data_dir, batch_idx=1, save_path=None):
    """
    Lorenz 96 모델의 느린 변수(X)에 대한 Hovmöller 다이어그램을 생성하는 함수
    
    Args:
        model_path: 학습된 모델 경로
        data_dir: 테스트 데이터가 저장된 디렉토리
        batch_idx: 사용할 배치 인덱스
        save_path: 그래프 저장 경로 (None인 경우 저장하지 않음)
    
    Returns:
        fig: 생성된 그래프 객체
    """
    # 모델 로드
    subgrid_nn = SubgridNN()
    model = NeuralLorenz96(subgrid_nn)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 평가 모드로 설정
    
    # 파라미터 설정
    K = 36  # 느린 변수(X)의 수
    F = 20  # 외부 강제력
    h = 1.0  # 커플링 계수
    b = 10   # 진폭 비율
    c = 10   # 시간 스케일 비율
    dt = 0.005  # 기본 시간 간격

    large_dt = 10*dt

    # 데이터 로드
    X_data = np.load(os.path.join(data_dir, f"X_batch_{batch_idx}.npy"))
    Y_data = np.load(os.path.join(data_dir, f"Y_batch_{batch_idx}.npy"))
    C_data = np.load(os.path.join(data_dir, f"C_batch_{batch_idx}.npy"))

    X_data = torch.from_numpy(X_data).float()
    Y_data = torch.from_numpy(Y_data).float()
    C_data = torch.from_numpy(C_data).float()
    
    # 입력 텐서의 shape이 [batch_size, timestep, features]인 경우 [timestep, features]로 차원 축소
    if len(X_data.shape) > 2:  # 3차원 텐서인 경우
        X_data = X_data[0]
        Y_data = Y_data[0]
        C_data = C_data[0]

    predicted_X, _ = model.integrate_L96_2t_with_neural_coupling(
        X_data,
        Y_data,
        F,
        h,
        b,
        c,
        dt=large_dt,
        neural_coupling=model.subgrid_nn(X_data)
    )
    
    # 10dt로 적분된 predicted_X에서 200 인덱스까지만 슬라이싱(=2000개의 데이터 포인트)
    predicted_X = predicted_X.detach().numpy()
    
    predicted_X = predicted_X[:200]

    # X_data에서 10 인덱스 간격으로 샘플링한 데이터 포인트 추출
    X_data_sampled = X_data[::10][:200]

    # 예측값과 샘플링된 실제 데이터의 차이 계산
    diff_X = X_data_sampled - predicted_X
    
    # 논문과 같은 스타일로 Hovmöller 다이어그램 생성
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # 컬러맵 설정
    cmap1 = plt.cm.viridis  # 상단 및 중간 패널용
    cmap2 = plt.cm.coolwarm  # 하단 패널용
        
    # 실제 X 값 (True X)
    im1 = axes[0].imshow(X_data.T, aspect='auto', cmap=cmap1, 
                         interpolation='none', origin='lower',
                         extent=[0, 10, 0, K], vmin=-10, vmax=10)
    axes[0].set_ylabel('True X')
    fig.colorbar(im1, ax=axes[0], orientation='vertical', pad=0.01)
    
    # 예측된 X 값 (Pred X̂)
    im2 = axes[1].imshow(predicted_X.T, aspect='auto', cmap=cmap1, 
                         interpolation='none', origin='lower',
                         extent=[0, 10, 0, K], vmin=-10, vmax=10)
    axes[1].set_ylabel('Pred X̂')
    fig.colorbar(im2, ax=axes[1], orientation='vertical', pad=0.01)
    
    # 차이 (X - X̂)
    # 차이의 최대 절대값을 기준으로 컬러맵 범위 설정
    max_diff = 20  # 논문과 같은 스케일 사용
    im3 = axes[2].imshow(diff_X.T, aspect='auto', cmap=cmap2, 
                         interpolation='none', origin='lower',
                         extent=[0, 10, 0, K],
                         vmin=-max_diff, vmax=max_diff)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('X - X̂')
    fig.colorbar(im3, ax=axes[2], orientation='vertical', pad=0.01)
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
#    plt.show()
    
    return fig


if __name__ == "__main__":
    # 파라미터 설정
    m_delta_t = 5
    batch_size = 100
    num_epoch = 2000

    # Kang's experimental setup
    K = 36  # Number of globa-scale variables X
    J = 10  # Number of local-scale Y variables per single global-scale X variable
    F = 20  # Forcing
    h = 1.0  # Coupling coefficient
    b = 10    # Ratio of amplitudes
    
    c = 10    # time-scale ratio 설정
    # 데이터 로드
    data_list = []
    for i in range(1, 301):
        X_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"X_batch_{i}.npy"))
        Y_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"Y_batch_{i}.npy"))
        C_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"C_batch_{i}.npy"))
        data_list.append([X_data, Y_data, C_data])

    model = NeuralLorenz96(SubgridNN())
    optimizer = torch.optim.Adam(model.subgrid_nn.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()        
    losses = []
    '''
    # 데이터를 batch_size 만큼 묶어서 학습
    for n in range(num_epoch):
        # batch_size가 100이므로, 0부터 300 사이에 있는 100개의 정수를 랜덤으로 추출
        sample_idx = np.random.choice(np.arange(300), size=100, replace=False)
        batch = []
        random_start_point = np.random.randint(0, len(X_data[0]) - m_delta_t)
        for idx in sample_idx:
            # 전체 데이터에서 해당 idx의 X_data, Y_data, t_data, C_data 데이터를 텐서로 변환
            X_data = torch.from_numpy(data_list[idx][0]).float()
            Y_data = torch.from_numpy(data_list[idx][1]).float()
            C_data = torch.from_numpy(data_list[idx][2]).float()

            # random_start_point 설정

            # ramdom_start_point 에서 m_delta_t 만큼의 데이터를 해당 idx의 데이터에서 추출
            sliced_X_data = X_data[:, random_start_point:random_start_point + m_delta_t, :]
            sliced_Y_data = Y_data[:, random_start_point:random_start_point + m_delta_t, :]
            sliced_C_data = C_data[:, random_start_point:random_start_point + m_delta_t, :]

            # batch 데이터 추가
            batch.append([sliced_X_data, sliced_Y_data, sliced_C_data])

        # batch 데이터를 텐서로 변환(batch_size, m_delta_t, X_dim)
        batch_X_data = torch.stack([batch[i][0] for i in range(batch_size)]).squeeze(1)
        batch_Y_data = torch.stack([batch[i][1] for i in range(batch_size)]).squeeze(1)
        batch_C_data = torch.stack([batch[i][2] for i in range(batch_size)]).squeeze(1)

        # 커플링 항 예측
        predicted_batch_C = model.subgrid_nn(batch_X_data)

        # batch의 개별 X, Y 데이터와 예측된 커플링 항을 이용해서 시스템 정수해(X, Y) 계산
        predicted_X_data = []
        predicted_Y_data = []

        for i in range(batch_size):
            predicted_X, predicted_Y = model.integrate_L96_2t_with_neural_coupling(batch_X_data[i], batch_Y_data[i], F, h, b, c, dt=0.005, neural_coupling=predicted_batch_C[i])
            predicted_X_data.append(predicted_X)
            predicted_Y_data.append(predicted_Y)

        loss = calculate_loss(batch_X_data, batch_Y_data, predicted_X_data, predicted_Y_data)
        
        # 오차 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 학습 과정 출력
        print(f"Epoch {n+1}/{num_epoch}, Loss: {loss.item()}")
        losses.append(loss.item())
    # 학습된 모델 저장
    os.makedirs(os.path.join(os.getcwd(), "baseline_models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "baseline_models", "subgrid_nn.pth"))   

    plt.title("Loss")
    plt.plot(losses)
    plt.savefig(os.path.join(os.getcwd(), "baseline_models", "loss.png"))
    plt.show()
    '''
    # 모델 및 데이터 경로 설정
    model_path = os.path.join(os.getcwd(), "baseline_models", "subgrid_nn.pth")
    data_dir = os.path.join(os.getcwd(), "simulated_data")
    
    # 랜덤하게 1개의 배치 선택
    batch_indices = np.random.choice(range(1, 301), size=1, replace=False)[0]
    save_path = os.path.join(os.getcwd(), f"coupling_comparison_{i}.png")
        # 커플링 항 비교 및 시각화
    compare_coupling_terms(model_path, data_dir, batch_indices=[i], 
                        time_steps=2000, save_path=save_path)

    # 10 MTU 단위로 Slow variable 예측 후 차이 시각화
    save_path = os.path.join(os.getcwd(), f"hovmoller_diagram_{i}.png")

    # Hovmöller 다이어그램 생성
    plot_hovmoller_diagram(model_path, data_dir, batch_idx=i, save_path=save_path)