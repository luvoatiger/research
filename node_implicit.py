"""
Baseline 모델(Learning subgrid-scale models with neural ordinary differential equations, Kim et al., 2023) 구현
"""
import os

# OpenMP 중복 라이브러리 로드 문제 해결
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import sys
import os
import pickle

# Add current directory to path for metrics import
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from metrics import ClimateMetrics
    CLIMATE_METRICS_AVAILABLE = True
    print("ClimateMetrics imported successfully")
except ImportError as e:
    print(f"Warning: Could not import ClimateMetrics: {e}")
    print("Climate metrics calculation will be skipped")
    CLIMATE_METRICS_AVAILABLE = False


class SubgridNN(torch.nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=8):
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


def calculate_climate_metrics_for_batch(batch_X_true, batch_Y_true, predicted_X_data, predicted_Y_data, dt=0.005):
    """
    배치 데이터에 대해 climate metrics를 계산하는 함수
    
    Args:
        batch_X_true: 실제 X 데이터 [batch_size, time_steps, K]
        batch_Y_true: 실제 Y 데이터 [batch_size, time_steps, K*J]
        predicted_X_data: 예측된 X 데이터 리스트
        predicted_Y_data: 예측된 Y 데이터 리스트
        dt: 시간 간격
        
    Returns:
        dict: climate metrics 결과
    """
    if not CLIMATE_METRICS_AVAILABLE:
        print("ClimateMetrics not available. Skipping climate metrics calculation.")
        return None
    
    print("\n=== Calculating Climate Metrics for Batch ===")
    
    try:
        climate_metrics = ClimateMetrics()
        
        # 배치 데이터를 numpy 배열로 변환하고 형태 맞추기
        batch_size = len(predicted_X_data)
        m_delta_t = len(predicted_X_data[0])
        
        # 예측 데이터를 텐서로 변환
        predicted_X_tensor = torch.stack(predicted_X_data).detach().numpy()  # [batch_size, time_steps, K]
        predicted_Y_tensor = torch.stack(predicted_Y_data).detach().numpy()  # [batch_size, time_steps, K*J]
        
        # 실제 데이터를 numpy로 변환
        X_true_np = batch_X_true.detach().numpy()  # [batch_size, time_steps, K]
        Y_true_np = batch_Y_true.detach().numpy()  # [batch_size, time_steps, K*J]
        
        # 각 배치에 대해 climate metrics 계산
        all_metrics = []
        
        for i in range(batch_size):
            # X 변수에 대한 metrics 계산
            X_pred_sample = predicted_X_tensor[i]  # [time_steps, K]
            X_true_sample = X_true_np[i]          # [time_steps, K]
            
            # Y 변수에 대한 metrics 계산 (첫 번째 그룹만 사용하여 차원 맞추기)
            Y_pred_sample = predicted_Y_tensor[i, :, :8]  # [time_steps, 8] (첫 번째 X에 대응하는 Y들)
            Y_true_sample = Y_true_np[i, :, :8]          # [time_steps, 8]
            
            # X 변수 metrics
            X_metrics = climate_metrics.calculate_all_metrics(X_pred_sample, X_true_sample, dt=dt)
            
            # Y 변수 metrics
            Y_metrics = climate_metrics.calculate_all_metrics(Y_pred_sample, Y_true_sample, dt=dt)
            
            # 통합 metrics
            combined_metrics = {
                'X_metrics': X_metrics,
                'Y_metrics': Y_metrics,
                'batch_index': i
            }
            all_metrics.append(combined_metrics)
        
        # 전체 배치에 대한 평균 metrics 계산
        avg_X_metrics = {}
        avg_Y_metrics = {}
        
        # X metrics 평균
        for key in all_metrics[0]['X_metrics'].keys():
            if isinstance(all_metrics[0]['X_metrics'][key], (int, float)):
                values = [m['X_metrics'][key] for m in all_metrics if m['X_metrics'][key] is not None]
                if values:
                    avg_X_metrics[key] = np.mean(values)
        
        # Y metrics 평균
        for key in all_metrics[0]['Y_metrics'].keys():
            if isinstance(all_metrics[0]['Y_metrics'][key], (int, float)):
                values = [m['Y_metrics'][key] for m in all_metrics if m['Y_metrics'][key] is not None]
                if values:
                    avg_Y_metrics[key] = np.mean(values)
        
        # 결과 출력
        print("\n--- AVERAGE CLIMATE METRICS (X variables) ---")
        climate_metrics.print_metrics_summary(avg_X_metrics)
        
        print("\n--- AVERAGE CLIMATE METRICS (Y variables) ---")
        climate_metrics.print_metrics_summary(avg_Y_metrics)
        
        # 결과 저장
        results = {
            'individual_metrics': all_metrics,
            'average_X_metrics': avg_X_metrics,
            'average_Y_metrics': avg_Y_metrics,
            'batch_size': batch_size,
            'time_steps': m_delta_t
        }
        
        print(f"\nClimate metrics calculated for {batch_size} batches with {m_delta_t} time steps")
        return results
        
    except Exception as e:
        print(f"Error calculating climate metrics: {e}")
        return None


def evaluate_model_performance_with_climate_metrics(model, test_data, num_test_samples=10, dt=0.005):
    """
    학습된 모델의 성능을 climate metrics를 포함하여 평가하는 함수
    
    Args:
        model: 학습된 NeuralLorenz96 모델
        test_data: 테스트 데이터 리스트
        num_test_samples: 테스트할 샘플 수
        dt: 시간 간격
        
    Returns:
        dict: 평가 결과
    """
    if not CLIMATE_METRICS_AVAILABLE:
        print("ClimateMetrics not available. Skipping evaluation.")
        return None
    
    print(f"\n=== Model Performance Evaluation with Climate Metrics ===")
    print(f"Testing on {num_test_samples} samples...")
    
    model.eval()
    all_test_metrics = []
    
    with torch.no_grad():
        for i in range(min(num_test_samples, len(test_data))):
            print(f"\n--- Test Sample {i+1}/{num_test_samples} ---")
            
            # 테스트 데이터 로드
            X_data = test_data[i][0]  # [1, time_steps, K]
            Y_data = test_data[i][1]  # [1, time_steps, K*J]
            C_data = test_data[i][2]  # [1, time_steps, K]
            
            # 예측
            predicted_X, predicted_Y = model.integrate_L96_2t_with_neural_coupling(
                X_data, Y_data, F, h, b, c, dt=dt, neural_coupling=C_data
            )
            
            # Climate metrics 계산
            sample_metrics = calculate_climate_metrics_for_batch(
                X_data, Y_data, [predicted_X], [predicted_Y], dt=dt
            )
            
            if sample_metrics:
                all_test_metrics.append(sample_metrics)
    
    # 전체 테스트 결과 요약
    if all_test_metrics:
        print(f"\n=== FINAL TEST RESULTS SUMMARY ===")
        print(f"Successfully evaluated {len(all_test_metrics)} test samples")
        
        # X 변수에 대한 전체 평균 metrics
        all_X_metrics = [m['average_X_metrics'] for m in all_test_metrics]
        final_X_metrics = {}
        
        for key in all_X_metrics[0].keys():
            values = [m[key] for m in all_X_metrics if key in m and m[key] is not None]
            if values:
                final_X_metrics[key] = np.mean(values)
        
        print("\n--- FINAL AVERAGE CLIMATE METRICS (X variables) ---")
        climate_metrics = ClimateMetrics()
        climate_metrics.print_metrics_summary(final_X_metrics)
        
        return {
            'test_samples': all_test_metrics,
            'final_X_metrics': final_X_metrics,
            'num_test_samples': len(all_test_metrics)
        }
    
    return None


def plot_climate_metrics_comparison(metrics_results, save_path=None):
    """
    Climate metrics 결과를 시각화하는 함수
    
    Args:
        metrics_results: calculate_climate_metrics_for_batch의 결과
        save_path: 저장할 파일 경로
    """
    if not CLIMATE_METRICS_AVAILABLE or metrics_results is None:
        print("Cannot plot climate metrics - data not available")
        return
    
    try:
        climate_metrics = ClimateMetrics()
        
        # X 변수 metrics 시각화
        if 'average_X_metrics' in metrics_results:
            X_metrics = metrics_results['average_X_metrics']
            if 'autocorr_lags_pred' in X_metrics and 'autocorr_values_pred' in X_metrics:
                # Autocorrelation 비교
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 3, 1)
                plt.plot(X_metrics['autocorr_lags_pred'], X_metrics['autocorr_values_pred'], 'r--', label='Prediction', linewidth=2)
                plt.plot(X_metrics['autocorr_lags_true'], X_metrics['autocorr_values_true'], 'b-', label='True', linewidth=2)
                plt.title('Autocorrelation Function (X variables)')
                plt.xlabel('Lag')
                plt.ylabel('Autocorrelation')
                plt.legend()
                plt.grid(True)
                
                # PDF 비교
                plt.subplot(1, 3, 2)
                plt.plot(X_metrics['pdf_bins_pred'], X_metrics['pdf_values_pred'], 'r--', label='Prediction', linewidth=2)
                plt.plot(X_metrics['pdf_bins_true'], X_metrics['pdf_values_true'], 'b-', label='True', linewidth=2)
                plt.title('Probability Density Function (X variables)')
                plt.xlabel('Value')
                plt.ylabel('Probability density')
                plt.legend()
                plt.grid(True)
                
                # Mean State Error over time
                plt.subplot(1, 3, 3)
                plt.plot(X_metrics['mean_state_error'], 'g-', linewidth=2)
                plt.title('Mean State Error Over Time (X variables)')
                plt.xlabel('Time step')
                plt.ylabel('Error')
                plt.grid(True)
                
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    print(f"Climate metrics comparison plot saved to {save_path}")
                
                plt.show()
        
    except Exception as e:
        print(f"Error plotting climate metrics: {e}")



if __name__ == "__main__":
    # 파라미터 설정
    m_delta_t = 5
    batch_size = 100
    num_epoch = 2000

    # Kang's experimental setup
    import json
    with open(os.path.join(os.getcwd(), "simulated_data", "metadata.json"), "r") as f:
        metadata = json.load(f)

    K = metadata["K"]  # Number of globa-scale variables X
    J = metadata["J"]  # Number of local-scale Y variables per single global-scale X variable
    F = metadata["F"]  # Forcing
    h = metadata["h"]  # Coupling coefficient
    b = metadata["b"]    # Ratio of amplitudes
    c = metadata["c"]    # time-scale ratio 설정

    # 데이터 로드
    data_list = []
    for i in range(1, 251):
        X_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"X_batch_coupled_{i}.npy"))
        Y_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"Y_batch_coupled_{i}.npy"))
        C_data = np.load(os.path.join(os.getcwd(), "simulated_data", f"C_batch_coupled_{i}.npy"))
        data_list.append([X_data, Y_data, C_data])

    model = NeuralLorenz96(SubgridNN())
    optimizer = torch.optim.Adam(model.subgrid_nn.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()        
    losses = []
    
    # 데이터를 batch_size 만큼 묶어서 학습
    for n in range(num_epoch):
        # batch_size가 100이므로, 0부터 300 사이에 있는 100개의 정수를 랜덤으로 추출
        sample_idx = np.random.choice(np.arange(300), size=100, replace=False)
        batch = []

        # random_start_point 설정
        random_start_point = np.random.randint(0, len(X_data[0]) - m_delta_t)
        for idx in sample_idx:
            # 전체 데이터에서 해당 idx의 X_data, Y_data, t_data, C_data 데이터를 텐서로 변환
            X_data = torch.from_numpy(data_list[idx][0]).float()
            Y_data = torch.from_numpy(data_list[idx][1]).float()
            C_data = torch.from_numpy(data_list[idx][2]).float()

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
    
    # 최종 모델 성능 평가 (climate metrics 포함)
    print("\n=== Final Model Performance Evaluation ===")
    try:
        # 테스트 데이터 준비 (학습에 사용하지 않은 데이터)
        test_data = data_list[251:301]  # 마지막 50개를 테스트용으로 사용
        
        final_evaluation = evaluate_model_performance_with_climate_metrics(
            model, test_data, num_test_samples=10, dt=0.005
        )
        
        if final_evaluation:
            # 최종 결과 시각화
            final_plot_path = os.path.join(os.getcwd(), "baseline_models", "final_climate_metrics.png")
            plot_climate_metrics_comparison(
                {'average_X_metrics': final_evaluation['final_X_metrics']}, 
                save_path=final_plot_path
            )
            
            # 최종 결과 저장
            final_metrics_path = os.path.join(os.getcwd(), "baseline_models", "final_climate_metrics.pkl")
            with open(final_metrics_path, 'wb') as f:
                pickle.dump(final_evaluation, f)
            print(f"Final evaluation results saved to {final_metrics_path}")
            
    except Exception as e:
        print(f"Error in final evaluation: {e}")
        print("Training completed without final evaluation")