"""
Baseline 모델(Learning subgrid-scale models with neural ordinary differential equations, Kim et al., 2023) 구현
"""
import os

import numpy as np
import torch


class SubgridNN(torch.nn.Module):
    def __init__(self, input_dim=36, hidden_dim=100, output_dim=36):
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
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float32, device=device)
        
        
        # X의 미분 계산 (PyTorch 연산 사용)
        Xdot = torch.roll(X, shifts=1, dims=0) * (torch.roll(X, shifts=-1, dims=0) - torch.roll(X, shifts=2, dims=0)) - X + F - neural_coupling
        
        # Y의 미분 계산
        # Y_plus_1, Y_plus_2, Y_minus_1 계산 (마지막 차원에 대해 roll)
        Y_plus_1 = torch.roll(Y, shifts=-1, dims=-1)
        Y_plus_2 = torch.roll(Y, shifts=-2, dims=-1)
        Y_minus_1 = torch.roll(Y, shifts=1, dims=-1)
        
        # X를 반복하여 Y와 같은 차원으로 확장
        X_repeated = torch.repeat_interleave(X, J, dim=-1)
          
        # Ydot 계산
        hcb = h * c / b
        Ydot = -c * b * Y_plus_1 * (Y_plus_2 - Y_minus_1) - c * Y + hcb * X_repeated
        
        return Xdot, Ydot, neural_coupling


    def integrate_L96_2t_with_neural_coupling(self, X0, Y0, F, h, b, c, dt=0.001, neural_coupling=None):
        """
        Integrates forward-in-time the two time-scale Lorenz 1996 model, using the RK4 integration method.
        Returns the full history with nt+1 values starting with initial conditions, X[:,0]=X0 and Y[:,0]=Y0,
        and ending with the final state, X[:,nt+1] and Y[:,nt+1] at time t0+nt*si.

        Note the model is intergrated

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


if __name__ == "__main__":
    # 파라미터 설정
    m_delta_t = 5
    batch_size = 100
    num_epoch = 2000

    # Kim's experimental setup
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

    # 데이터를 batch_size 만큼 묶어서 학습
    for n in range(num_epoch):
        # batch_size가 100이므로, 0부터 300 사이에 있는 100개의 정수를 랜덤으로 추출
        sample_idx = np.random.choice(np.arange(300), size=100, replace=False)
        batch = []
        for idx in sample_idx:
            # 전체 데이터에서 해당 idx의 X_data, Y_data, t_data, C_data 데이터를 텐서로 변환
            X_data = torch.from_numpy(data_list[idx][0]).float()
            Y_data = torch.from_numpy(data_list[idx][1]).float()
            C_data = torch.from_numpy(data_list[idx][2]).float()

            # random_start_point 설정
            random_start_point = np.random.randint(0, len(X_data[0]) - m_delta_t)

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

        # 계산된 시스템 정수해와 실제 시스템 정수해의 차이 계산
        loss = criterion(torch.stack(predicted_X_data), batch_X_data) + criterion(torch.stack(predicted_Y_data), batch_Y_data)

        # 오차 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 학습 과정 출력
        print(f"Epoch {n+1}/{num_epoch}, Loss: {loss.item()}")

    # 학습된 모델 저장
    os.makedirs(os.path.join(os.getcwd(), "baseline_models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "baseline_models", "subgrid_nn.pth"))   
