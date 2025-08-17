import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from metrics import ClimateMetrics

def lorenz96_system_equations(t, state, F, h, b, c, K, J):
    """
    Lorenz 96 two-time-scale 시스템의 미분 방정식
    
    Args:
        t: 시간
        state: 상태 벡터 [X, Y] (X는 K차원, Y는 K*J차원)
        F, h, b, c: 시스템 파라미터
        K: X 변수의 수
        J: 각 X에 대응하는 Y 변수의 수
    
    Returns:
        derivatives: 미분값 [dX/dt, dY/dt]
    """
    X = state[:K]
    Y = state[K:]
    
    # X의 미분 계산
    dX = np.zeros(K)
    for k in range(K):
        dX[k] = (X[(k+1) % K] - X[(k-2) % K]) * X[(k-1) % K] - X[k] + F - (h * c / b) * np.sum(Y[k*J:(k+1)*J])
    
    # Y의 미분 계산
    dY = np.zeros(K * J)
    for k in range(K):
        for j in range(J):
            idx = k * J + j
            dY[idx] = -b * c * Y[(idx + 1) % (K * J)] * (Y[(idx + 2) % (K * J)] - Y[(idx - 1) % (K * J)]) - c * Y[idx] + (h * c / b) * X[k]
    
    return np.concatenate([dX, dY])


def simulate_and_plot_lorenz96_x1_prediction(model, metadata, memory_length_TM, X_init, Y_init, t_end=10, t_start_plot=2, delta=0.005):
    """
    Lorenz 96 시스템의 첫 번째 변수(X1)에 대한 모델 예측과 실제 시뮬레이션을 비교하여 시각화하는 함수

    Args:
        model: 학습된 dAMZ 모델
        metadata: Lorenz 96 시스템의 메타데이터
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
    """
    # t_end가 t_start_plot보다 큰지 확인하고 예외 처리
    if t_end <= t_start_plot:
        print(f"경고: t_end({t_end})가 t_start_plot({t_start_plot})보다 작거나 같습니다.")
        print(f"t_end를 {t_start_plot + 1.0}로 조정합니다.")
        t_end = t_start_plot + 1.0
    
    # 시스템 파라미터 추출
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    
    # 초기 조건 설정 (Lorenz 96 시스템에 맞는 초기 조건)
    k = np.arange(K)
    j = np.arange(J * K)
    
    # 초기 조건 함수 (multiscale_lorenz.py에서 가져옴)
    def s(k, K):
        return 2 * np.pi * k / K
    
    if X_init is None or Y_init is None:
        X_init = s(k, K) * (s(k, K) - 1) * (s(k, K) + 1)
        Y_init = 0 * s(j, J * K) * (s(j, J * K) - 1) * (s(j, J * K) + 1)
    initial_condition = np.concatenate([X_init, Y_init])

    print(f"초기 조건 X: {X_init}")
    print(f"초기 조건 Y: {Y_init[:J]} (첫 번째 그룹만 표시)")

    # 실제 시뮬레이션
    t_eval = np.arange(0, t_end, delta)  # t_end는 포함하지 않음
    t_span = (0, t_end)

    try:
        # Lorenz 96 시스템 적분
        sol = solve_ivp(
            fun=lambda t, y: lorenz96_system_equations(t, y, F, h, b, c, K, J),
            t_span=t_span,
            y0=initial_condition,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )

        t_vals = sol.t
        X_true = sol.y[:K, :].T  # X 변수들만 추출
        x1_true = X_true[:, 0]  # 첫 번째 변수

        print(f"시뮬레이션 완료: {len(t_vals)} 시간 스텝")
        print(f"X1 범위: [{x1_true.min():.3f}, {x1_true.max():.3f}]")

        # 시스템 발산 감지
        if np.any(np.abs(x1_true) > 1000) or len(t_vals) < t_end / delta * 0.5:
            print("경고: 시스템이 발산했습니다. 이 초기 조건은 건너뜁니다.")
            return

        # 모델 예측
        model.eval()
        n_M = int(memory_length_TM / dt)  # 메모리 길이 (일관성을 위해 동일하게 설정)

        # HistoryBuffer 초기화
        history_buffer = HistoryBuffer(n_M + 1, K)
        
        # 초기 히스토리 채우기
        for i in range(n_M + 1):
            history_buffer.update(torch.FloatTensor(X_true[i]))

        # 메모리 항목들을 포함한 입력 데이터 준비
        x1_pred = []
        with torch.no_grad():
            for i in range(n_M + 1, len(X_true)):
                # HistoryBuffer에서 히스토리 가져오기
                history = history_buffer.get_history()  # [n_M+1, K]
                Z_input = history.reshape(-1)  # [D]
                Z_tensor = Z_input.unsqueeze(0)  # [1, D]

                # 예측
                z_pred = model(Z_tensor, enable_dropout=False)
                x1_pred.append(z_pred[0, 0].item())  # X1 예측값
                
                # 새로운 상태로 히스토리 업데이트
                history_buffer.update(torch.FloatTensor(z_pred[0]))

        print(f"예측 완료: {len(x1_pred)} 개의 예측값")

        # 시각화
        t_pred = t_vals[n_M + 1:]
        mask = (t_pred >= t_start_plot)
        t_plot = t_pred[mask]
        x1_true_plot = x1_true[n_M + 1:][mask]
        x1_pred_plot = np.array(x1_pred)[mask]

        print(f"필터링 후 데이터: {len(t_plot)} 개의 시간 포인트")
        print(f"t_plot 범위: [{t_plot.min():.3f}, {t_plot.max():.3f}]")

        if len(t_plot) == 0:
            print("경고: 필터링 후 데이터가 없습니다!")
            print(f"t_start_plot={t_start_plot}, t_pred 범위=[{t_pred.min():.3f}, {t_pred.max():.3f}]")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(t_plot, x1_true_plot, 'b-', label='True $X_1(t)$', linewidth=2)
        plt.plot(t_plot, x1_pred_plot, 'r--', label='Pred $X_1(t)$', linewidth=2)
        plt.xlabel('time t')
        plt.ylabel('$X_1(t)$')
        plt.title(f'Lorenz 96 dAMZ prediction vs real simulation [{t_start_plot}, {t_end}]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # MSE 및 RMSE 계산
        mse = np.mean((x1_true_plot - x1_pred_plot) ** 2)
        rmse = np.sqrt(mse)
        print(f'예측 MSE: {mse:.6f}')
        print(f'예측 RMSE: {rmse:.6f}')
        
        # 데이터 범위 대비 상대적 오차
        data_range = x1_true_plot.max() - x1_true_plot.min()
        relative_error = rmse / data_range * 100
        print(f'상대적 오차: {relative_error:.2f}%')
        
        # Climate Metrics 계산 추가
        print("\n[+] Climate Metrics 계산...")
        try:
            # 데이터 형태 맞추기 (1차원 -> 2차원)
            u_true_1d = x1_true_plot.reshape(-1, 1)
            u_pred_1d = x1_pred_plot.reshape(-1, 1)
            
            # Climate Metrics 계산
            climate_metrics = ClimateMetrics()
            metrics = climate_metrics.calculate_all_metrics(u_pred_1d, u_true_1d, dt=delta)
            
            # 결과 출력
            climate_metrics.print_metrics_summary(metrics)
            
            # 시각화 (선택사항)
            save_path = f"climate_metrics_x1_{t_start_plot}_{t_end}.png"
            climate_metrics.plot_metrics_over_time(metrics, t_plot, save_path)
            
        except Exception as e:
            print(f"Climate Metrics 계산 중 오류 발생: {e}")

    except Exception as e:
        print(f"시뮬레이션 중 오류 발생: {e}")
        print(f"초기 조건에서 시스템이 불안정할 수 있습니다.")


def simulate_and_plot_lorenz96_all_variables_prediction(model, metadata, memory_length_TM, X_init, Y_init, t_end=10, t_start_plot=2, delta=0.005):
    """
    Lorenz 96 시스템의 모든 X 변수에 대한 모델 예측과 실제 시뮬레이션을 비교하여 시각화하는 함수

    Args:
        model: 학습된 dAMZ 모델
        metadata: Lorenz 96 시스템의 메타데이터
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
    """
    # t_end가 t_start_plot보다 큰지 확인하고 예외 처리
    if t_end <= t_start_plot:
        print(f"경고: t_end({t_end})가 t_start_plot({t_start_plot})보다 작거나 같습니다.")
        print(f"t_end를 {t_start_plot + 1.0}로 조정합니다.")
        t_end = t_start_plot + 1.0
    
    # 시스템 파라미터 추출
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    
    # 초기 조건 설정
    k = np.arange(K)
    j = np.arange(J * K)
    # 초기 조건 함수 (multiscale_lorenz.py에서 가져옴)
    def s(k, K):
        return 2 * np.pi * k / K

    if X_init is None or Y_init is None:
        X_init = s(k, K) * (s(k, K) - 1) * (s(k, K) + 1)
        Y_init = 0 * s(j, J * K) * (s(j, J * K) - 1) * (s(j, J * K) + 1)
        
    initial_condition = np.concatenate([X_init, Y_init])

    # 실제 시뮬레이션
    t_eval = np.arange(0, t_end, delta)  # t_end는 포함하지 않음
    t_span = (0, t_end)

    try:
        sol = solve_ivp(
            fun=lambda t, y: lorenz96_system_equations(t, y, F, h, b, c, K, J),
            t_span=t_span,
            y0=initial_condition,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )

        t_vals = sol.t
        X_true = sol.y[:K, :].T

        # 모델 예측
        model.eval()
        n_M = int(memory_length_TM / dt)  # 메모리 길이 (일관성을 위해 동일하게 설정)

        # HistoryBuffer 초기화
        history_buffer = HistoryBuffer(n_M + 1, K)
        
        # 초기 히스토리 채우기
        for i in range(n_M + 1):
            history_buffer.update(torch.FloatTensor(X_true[i]))

        X_pred = []
        with torch.no_grad():
            for i in range(n_M + 1, len(X_true)):
                # HistoryBuffer에서 히스토리 가져오기
                history = history_buffer.get_history()  # [n_M+1, K]
                Z_input = history.reshape(-1)  # [D]
                Z_tensor = Z_input.unsqueeze(0)  # [1, D]
                z_pred = model(Z_tensor)
                X_pred.append(z_pred[0].numpy())
                
                # 새로운 상태로 히스토리 업데이트
                history_buffer.update(torch.FloatTensor(z_pred[0]))

        X_pred = np.array(X_pred)
        t_pred = t_vals[n_M + 1:]
        mask = (t_pred >= t_start_plot)
        t_plot = t_pred[mask]
        X_true_plot = X_true[n_M + 1:][mask]
        X_pred_plot = X_pred[mask]

        # 모든 변수 시각화
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for i in range(K):
            ax = axes[i]
            ax.plot(t_plot, X_true_plot[:, i], 'b-', label='True', linewidth=2)
            ax.plot(t_plot, X_pred_plot[:, i], 'r--', label='Pred', linewidth=2)
            ax.set_xlabel('time t')
            ax.set_ylabel(f'$X_{i+1}(t)$')
            ax.set_title(f'Variable {i+1}')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

        # 전체 MSE 및 RMSE 계산
        mse_total = np.mean((X_true_plot - X_pred_plot) ** 2)
        rmse_total = np.sqrt(mse_total)
        print(f'전체 예측 MSE: {mse_total:.6f}')
        print(f'전체 예측 RMSE: {rmse_total:.6f}')
        
        # 각 변수별 MSE 계산
        for i in range(K):
            mse_var = np.mean((X_true_plot[:, i] - X_pred_plot[:, i]) ** 2)
            rmse_var = np.sqrt(mse_var)
            print(f'변수 {i+1} MSE: {mse_var:.6f}, RMSE: {rmse_var:.6f}')
        
        # Climate Metrics 계산 추가
        print("\n[+] Climate Metrics 계산 (모든 변수)...")
        try:
            # Climate Metrics 계산
            climate_metrics = ClimateMetrics()
            metrics = climate_metrics.calculate_all_metrics(X_pred_plot, X_true_plot, dt=delta)
            
            # 결과 출력
            climate_metrics.print_metrics_summary(metrics)
            
            # 시각화 (선택사항)
            save_path = f"climate_metrics_all_vars_{t_start_plot}_{t_end}.png"
            climate_metrics.plot_metrics_over_time(metrics, t_plot, save_path)
            
        except Exception as e:
            print(f"Climate Metrics 계산 중 오류 발생: {e}")

    except Exception as e:
        print(f"시뮬레이션 중 오류 발생: {e}")


def simulate_and_plot_lorenz96_x1_prediction_with_uncertainty(model, metadata, memory_length_TM, X_init, Y_init, t_end=10, t_start_plot=2, delta=0.005, num_mc_samples=100):
    """
    Lorenz 96 시스템의 첫 번째 변수(X1)에 대한 불확실성을 포함한 모델 예측과 실제 시뮬레이션을 비교하여 시각화하는 함수

    Args:
        model: 학습된 dAMZ 모델 (Monte Carlo Dropout 포함)
        metadata: Lorenz 96 시스템의 메타데이터
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
        num_mc_samples: Monte Carlo 샘플 수
    """
    # t_end가 t_start_plot보다 큰지 확인하고 예외 처리
    if t_end <= t_start_plot:
        print(f"경고: t_end({t_end})가 t_start_plot({t_start_plot})보다 작거나 같습니다.")
        print(f"t_end를 {t_start_plot + 1.0}로 조정합니다.")
        t_end = t_start_plot + 1.0
    
    # 시스템 파라미터 추출
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    
    # 초기 조건 설정 (Lorenz 96 시스템에 맞는 초기 조건)
    k = np.arange(K)
    j = np.arange(J * K)
    def s(k, K):
        return 2 * np.pi * k / K

    if X_init is None or Y_init is None:
        X_init = s(k, K) * (s(k, K) - 1) * (s(k, K) + 1)
        Y_init = 0 * s(j, J * K) * (s(j, J * K) - 1) * (s(j, J * K) + 1)
    
    initial_condition = np.concatenate([X_init, Y_init])
    print(f"초기 조건 X: {X_init}")
    print(f"초기 조건 Y: {Y_init[:J]} (첫 번째 그룹만 표시)")

    # 실제 시뮬레이션
    t_eval = np.arange(0, t_end, delta)  # t_end는 포함하지 않음
    t_span = (0, t_end)

    try:
        # Lorenz 96 시스템 적분
        sol = solve_ivp(
            fun=lambda t, y: lorenz96_system_equations(t, y, F, h, b, c, K, J),
            t_span=t_span,
            y0=initial_condition,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )

        t_vals = sol.t
        X_true = sol.y[:K, :].T  # X 변수들만 추출
        x1_true = X_true[:, 0]  # 첫 번째 변수

        print(f"시뮬레이션 완료: {len(t_vals)} 시간 스텝")
        print(f"X1 범위: [{x1_true.min():.3f}, {x1_true.max():.3f}]")

        # 시스템 발산 감지
        if np.any(np.abs(x1_true) > 1000) or len(t_vals) < t_end / delta * 0.5:
            print("경고: 시스템이 발산했습니다. 이 초기 조건은 건너뜁니다.")
            return

        # 모델 예측 (불확실성 포함)
        n_M = int(memory_length_TM / dt)  # 메모리 길이 (일관성을 위해 동일하게 설정)

        # HistoryBuffer 초기화
        history_buffer = HistoryBuffer(n_M + 1, K)
        
        # 초기 히스토리 채우기
        for i in range(n_M + 1):
            history_buffer.update(torch.FloatTensor(X_true[i]))

        # 메모리 항목들을 포함한 입력 데이터 준비
        x1_pred_mean = []
        x1_pred_std = []
        
        with torch.no_grad():
            for i in range(n_M + 1, len(X_true)):
                # HistoryBuffer에서 히스토리 가져오기
                history = history_buffer.get_history()  # [n_M+1, K]
                Z_input = history.reshape(-1)  # [D]
                Z_tensor = Z_input.unsqueeze(0)  # [1, D]

                # Monte Carlo Dropout을 사용한 불확실성 추정
                mean_pred, std_pred = model.predict_with_uncertainty(Z_tensor, num_samples=num_mc_samples)
                x1_pred_mean.append(mean_pred[0, 0].item())  # X1 평균 예측값
                x1_pred_std.append(std_pred[0, 0].item())    # X1 표준편차
                
                # 새로운 상태로 히스토리 업데이트
                history_buffer.update(torch.FloatTensor(z_pred[0]))

        print(f"예측 완료: {len(x1_pred_mean)} 개의 예측값 (Monte Carlo 샘플 수: {num_mc_samples})")

        # 시각화 - 예측값과 실제값이 같은 시점에서 시작하도록 수정
        t_pred = t_vals[n_M + 1:]
        mask = (t_pred >= t_start_plot)
        t_plot = t_pred[mask]
        x1_true_plot = x1_true[n_M + 1:][mask]  # 예측 가능한 시점부터의 실제값
        x1_pred_mean_plot = np.array(x1_pred_mean)[mask]
        x1_pred_std_plot = np.array(x1_pred_std)[mask]

        print(f"필터링 후 데이터: {len(t_plot)} 개의 시간 포인트")
        print(f"t_plot 범위: [{t_plot.min():.3f}, {t_plot.max():.3f}]")

        if len(t_plot) == 0:
            print("경고: 필터링 후 데이터가 없습니다!")
            print(f"t_start_plot={t_start_plot}, t_pred 범위=[{t_pred.min():.3f}, {t_pred.max():.3f}]")
            return

        # 불확실성을 포함한 시각화
        plt.figure(figsize=(12, 8))
        
        # 실제 값과 평균 예측
        plt.plot(t_plot, x1_true_plot, 'b-', label='True $X_1(t)$', linewidth=2)
        plt.plot(t_plot, x1_pred_mean_plot, 'r--', label='Pred $X_1(t)$ (mean)', linewidth=2)
        
        # 불확실성 구간 (95% 신뢰구간)
        plt.fill_between(t_plot, 
                        x1_pred_mean_plot - 2*x1_pred_std_plot, 
                        x1_pred_mean_plot + 2*x1_pred_std_plot, 
                        alpha=0.1, color='red', label='95% Confidence Interval')
        
        # 1 표준편차 구간
        plt.fill_between(t_plot, 
                        x1_pred_mean_plot - x1_pred_std_plot, 
                        x1_pred_mean_plot + x1_pred_std_plot, 
                        alpha=0.1, color='red', label='±1σ Interval')
        
        plt.xlabel('time t')
        plt.ylabel('$X_1(t)$')
        plt.title(f'Lorenz 96 dAMZ prediction with uncertainty [{t_start_plot}, {t_end}]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # MSE 및 RMSE 계산 (평균 예측 기준)
        mse = np.mean((x1_true_plot - x1_pred_mean_plot) ** 2)
        rmse = np.sqrt(mse)
        print(f'예측 MSE: {mse:.6f}')
        print(f'예측 RMSE: {rmse:.6f}')
        
        # 데이터 범위 대비 상대적 오차
        data_range = x1_true_plot.max() - x1_true_plot.min()
        relative_error = rmse / data_range * 100
        print(f'상대적 오차: {relative_error:.2f}%')
        
        # 불확실성 통계
        mean_uncertainty = np.mean(x1_pred_std_plot)
        max_uncertainty = np.max(x1_pred_std_plot)
        print(f'평균 불확실성 (표준편차): {mean_uncertainty:.6f}')
        print(f'최대 불확실성 (표준편차): {max_uncertainty:.6f}')
        
        # Climate Metrics 계산 추가 (불확실성 포함)
        print("\n[+] Climate Metrics 계산 (불확실성 포함)...")
        try:
            # 데이터 형태 맞추기 (1차원 -> 2차원)
            u_true_1d = x1_true_plot.reshape(-1, 1)
            u_pred_1d = x1_pred_mean_plot.reshape(-1, 1)
            
            # Climate Metrics 계산
            climate_metrics = ClimateMetrics()
            metrics = climate_metrics.calculate_all_metrics(u_pred_1d, u_true_1d, dt=delta)
            
            # 결과 출력
            climate_metrics.print_metrics_summary(metrics)
            
            # 시각화 (선택사항)
            save_path = f"climate_metrics_x1_uncertainty_{t_start_plot}_{t_end}.png"
            climate_metrics.plot_metrics_over_time(metrics, t_plot, save_path)
            
        except Exception as e:
            print(f"Climate Metrics 계산 중 오류 발생: {e}")

    except Exception as e:
        print(f"시뮬레이션 중 오류 발생: {e}")
        print(f"초기 조건에서 시스템이 불안정할 수 있습니다.")


def simulate_and_plot_lorenz96_all_variables_prediction_with_uncertainty(model, metadata, memory_length_TM, X_init, Y_init, t_end=10, t_start_plot=2, delta=0.005, num_mc_samples=100):
    """
    Lorenz 96 시스템의 모든 X 변수에 대한 불확실성을 포함한 모델 예측과 실제 시뮬레이션을 비교하여 시각화하는 함수

    Args:
        model: 학습된 dAMZ 모델 (Monte Carlo Dropout 포함)
        metadata: Lorenz 96 시스템의 메타데이터
        X_init: X 변수들의 초기 조건
        Y_init: Y 변수들의 초기 조건
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
        num_mc_samples: Monte Carlo 샘플 수
    """
    # t_end가 t_start_plot보다 큰지 확인하고 예외 처리
    if t_end <= t_start_plot:
        print(f"경고: t_end({t_end})가 t_start_plot({t_start_plot})보다 작거나 같습니다.")
        print(f"t_end를 {t_start_plot + 1.0}로 조정합니다.")
        t_end = t_start_plot + 1.0
    
    # 시스템 파라미터 추출
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    
    # 초기 조건 설정
    k = np.arange(K)
    j = np.arange(J * K)
    # 초기 조건 함수 (multiscale_lorenz.py에서 가져옴)
    def s(k, K):
        return 2 * np.pi * k / K

    if X_init is None or Y_init is None:
        X_init = s(k, K) * (s(k, K) - 1) * (s(k, K) + 1)
        Y_init = 0 * s(j, J * K) * (s(j, J * K) - 1) * (s(j, J * K) + 1)

    initial_condition = np.concatenate([X_init, Y_init])
    print(f"초기 조건 X: {X_init}")
    print(f"초기 조건 Y: {Y_init[:J]} (첫 번째 그룹만 표시)")

    # 실제 시뮬레이션
    t_eval = np.arange(0, t_end, delta)  # t_end는 포함하지 않음
    t_span = (0, t_end)

    try:
        sol = solve_ivp(
            fun=lambda t, y: lorenz96_system_equations(t, y, F, h, b, c, K, J),
            t_span=t_span,
            y0=initial_condition,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-9,
            atol=1e-11
        )

        t_vals = sol.t
        X_true = sol.y[:K, :].T

        # 시스템 발산 감지
        if np.any(np.abs(X_true) > 1000) or len(t_vals) < t_end / delta * 0.5:
            print("경고: 시스템이 발산했습니다. 이 초기 조건은 건너뜁니다.")
            return

        # 모델 예측 (불확실성 포함)
        n_M = int(memory_length_TM / dt)  # 메모리 길이 (일관성을 위해 동일하게 설정)

        # HistoryBuffer 초기화
        history_buffer = HistoryBuffer(n_M + 1, K)
        
        # 초기 히스토리 채우기
        for i in range(n_M + 1):
            history_buffer.update(torch.FloatTensor(X_true[i]))

        X_pred_mean = []
        X_pred_std = []
        
        with torch.no_grad():
            for i in range(n_M + 1, len(X_true)):
                # HistoryBuffer에서 히스토리 가져오기
                history = history_buffer.get_history()  # [n_M+1, K]
                Z_input = history.reshape(-1)  # [D]
                Z_tensor = Z_input.unsqueeze(0)  # [1, D]
                
                # Monte Carlo Dropout을 사용한 불확실성 추정
                mean_pred, std_pred = model.predict_with_uncertainty(Z_tensor, num_samples=num_mc_samples)
                X_pred_mean.append(mean_pred[0].numpy())
                X_pred_std.append(std_pred[0].numpy())
                
                # 새로운 상태로 히스토리 업데이트
                history_buffer.update(torch.FloatTensor(z_pred[0]))

            X_pred_mean = np.array(X_pred_mean)
            X_pred_std = np.array(X_pred_std)
            t_pred = t_vals[n_M + 1:]
            mask = (t_pred >= t_start_plot)
            t_plot = t_pred[mask]
            X_true_plot = X_true[n_M + 1:][mask]
            X_pred_mean_plot = X_pred_mean[mask]
            X_pred_std_plot = X_pred_std[mask]

        print(f"예측 완료: {len(X_pred_mean)} 개의 예측값 (Monte Carlo 샘플 수: {num_mc_samples})")

        # 모든 변수 시각화 (불확실성 포함)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i in range(K):
            ax = axes[i]
            
            # 실제 값과 평균 예측
            ax.plot(t_plot, X_true_plot[:, i], 'b-', label='True', linewidth=2)
            ax.plot(t_plot, X_pred_mean_plot[:, i], 'r--', label='Pred (mean)', linewidth=2)
            
            # 불확실성 구간 (95% 신뢰구간)
            ax.fill_between(t_plot, 
                           X_pred_mean_plot[:, i] - 2*X_pred_std_plot[:, i], 
                           X_pred_mean_plot[:, i] + 2*X_pred_std_plot[:, i], 
                           alpha=0.1, color='red', label='95% Confidence')
            
            # 1 표준편차 구간
            ax.fill_between(t_plot, 
                           X_pred_mean_plot[:, i] - X_pred_std_plot[:, i], 
                           X_pred_mean_plot[:, i] + X_pred_std_plot[:, i], 
                           alpha=0.2, color='red', label='±1σ')
            
            ax.set_xlabel('time t')
            ax.set_ylabel(f'$X_{i+1}(t)$')
            ax.set_title(f'Variable {i+1} with Uncertainty')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

        # 전체 MSE 및 RMSE 계산 (평균 예측 기준)
        mse_total = np.mean((X_true_plot - X_pred_mean_plot) ** 2)
        rmse_total = np.sqrt(mse_total)
        print(f'전체 예측 MSE: {mse_total:.6f}')
        print(f'전체 예측 RMSE: {rmse_total:.6f}')
        
        # 각 변수별 MSE 계산
        for i in range(K):
            mse_var = np.mean((X_true_plot[:, i] - X_pred_mean_plot[:, i]) ** 2)
            rmse_var = np.sqrt(mse_var)
            print(f'변수 {i+1} MSE: {mse_var:.6f}, RMSE: {rmse_var:.6f}')
        
        # 불확실성 통계
        mean_uncertainty = np.mean(X_pred_std_plot)
        max_uncertainty = np.max(X_pred_std_plot)
        print(f'평균 불확실성 (표준편차): {mean_uncertainty:.6f}')
        print(f'최대 불확실성 (표준편차): {max_uncertainty:.6f}')
        
        # 각 변수별 불확실성
        for i in range(K):
            var_mean_uncertainty = np.mean(X_pred_std_plot[:, i])
            var_max_uncertainty = np.max(X_pred_std_plot[:, i])
        
        # Climate Metrics 계산 추가 (불확실성 포함, 모든 변수)
        print("\n[+] Climate Metrics 계산 (불확실성 포함, 모든 변수)...")
        try:
            # Climate Metrics 계산
            climate_metrics = ClimateMetrics()
            metrics = climate_metrics.calculate_all_metrics(X_pred_mean_plot, X_true_plot, dt=delta)
            
            # 결과 출력
            climate_metrics.print_metrics_summary(metrics)
            
            # 시각화 (선택사항)
            save_path = f"climate_metrics_all_vars_uncertainty_{t_start_plot}_{t_end}.png"
            climate_metrics.plot_metrics_over_time(metrics, t_plot, save_path)
            
        except Exception as e:
            print(f"Climate Metrics 계산 중 오류 발생: {e}")
            print(f'변수 {i+1} 평균 불확실성: {var_mean_uncertainty:.6f}, 최대 불확실성: {var_max_uncertainty:.6f}')

    except Exception as e:
        print(f"시뮬레이션 중 오류 발생: {e}")
        print(f"초기 조건에서 시스템이 불안정할 수 있습니다.")


def evaluate_extrapolation_performance(model, metadata, memory_length_TM, num_trials=5, t_end=10, t_start_plot=2, delta=0.005):
    """
    랜덤 초기 조건에 대해 extrapolation 성능을 평가하는 함수
    논문의 성능 측정 방법: np.linalg.norm(X_ext-Xpred_ext) / np.linalg.norm(X_ext)
    
    Args:
        model: 학습된 dAMZ 모델
        metadata: Lorenz 96 시스템의 메타데이터
        num_trials: 평가할 랜덤 초기 조건의 수
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
    
    Returns:
        performance_scores: 각 시도별 성능 점수 리스트
        mean_performance: 평균 성능 점수
    """
    # t_end가 t_start_plot보다 큰지 확인하고 예외 처리
    if t_end <= t_start_plot:
        print(f"경고: t_end({t_end})가 t_start_plot({t_start_plot})보다 작거나 같습니다.")
        print(f"t_end를 {t_start_plot + 1.0}로 조정합니다.")
        t_end = t_start_plot + 1.0
    
    # 시스템 파라미터 추출
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    
    performance_scores = []
    
    print(f"\n=== Extrapolation 성능 평가 (랜덤 초기 조건 {num_trials}개) ===")
    
    for trial in range(num_trials):
        print(f"\n--- 시도 {trial + 1}/{num_trials} ---")
        
        # 랜덤 초기 조건 생성
        np.random.seed(42 + trial)  # 재현성을 위한 시드 설정
        
        # X 초기 조건: 논문과 유사한 범위로 설정
        X_init = np.random.uniform(-5, 5, K)
        
        # Y 초기 조건: 작은 값으로 설정
        Y_init = np.random.uniform(-0.1, 0.1, K * J)
        
        initial_condition = np.concatenate([X_init, Y_init])
        print(f"초기 조건 X: {X_init}")
        print(f"초기 조건 Y 범위: [{Y_init.min():.3f}, {Y_init.max():.3f}]")

        # 실제 시뮬레이션
        t_eval = np.arange(0, t_end, delta)  # t_end는 포함하지 않음
        t_span = (0, t_end)

        try:
            # Lorenz 96 시스템 적분
            sol = solve_ivp(
                fun=lambda t, y: lorenz96_system_equations(t, y, F, h, b, c, K, J),
                t_span=t_span,
                y0=initial_condition,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-9,
                atol=1e-11
            )

            t_vals = sol.t
            X_true = sol.y[:K, :].T  # X 변수들만 추출

            # 시스템 발산 감지
            if np.any(np.abs(X_true) > 1000) or len(t_vals) < t_end / delta * 0.5:
                print("경고: 시스템이 발산했습니다. 이 시도는 건너뜁니다.")
                continue

            # 모델 예측
            model.eval()
            n_M = int(memory_length_TM / dt)  # 메모리 길이 (일관성을 위해 동일하게 설정)

            X_pred = []
            with torch.no_grad():
                for i in range(n_M + 1, len(X_true)):
                    # Z = (z_n^T, z_{n-1}^T, ..., z_{n-nM}^T)^T
                    Z_input = X_true[i-n_M-1:i].reshape(-1)  # [D]
                    Z_tensor = torch.FloatTensor(Z_input).unsqueeze(0)  # [1, D]

                    # 예측
                    z_pred = model(Z_tensor)
                    X_pred.append(z_pred[0].numpy())

            X_pred = np.array(X_pred)
            t_pred = t_vals[n_M + 1:]
            
            # extrapolation 구간 설정 (t_start_plot 이후)
            mask = (t_pred >= t_start_plot)
            X_ext = X_true[n_M + 1:][mask]  # 실제 extrapolation 데이터
            Xpred_ext = X_pred[mask]  # 예측 extrapolation 데이터

            if len(X_ext) == 0:
                print("경고: extrapolation 데이터가 없습니다!")
                continue

            # 논문의 성능 측정 방법 적용
            # np.linalg.norm(X_ext-Xpred_ext) / np.linalg.norm(X_ext)
            numerator = np.linalg.norm(X_ext - Xpred_ext)
            denominator = np.linalg.norm(X_ext)
            
            if denominator == 0:
                print("경고: 분모가 0입니다!")
                continue
                
            performance_score = numerator / denominator
            performance_scores.append(performance_score)
            
            print(f"Extrapolation 성능 점수: {performance_score:.6f}")
            print(f"  - 분자 (예측 오차 norm): {numerator:.6f}")
            print(f"  - 분모 (실제 데이터 norm): {denominator:.6f}")
            print(f"  - 데이터 포인트 수: {len(X_ext)}")
            
            # Climate Metrics 계산 추가
            try:
                print("  - Climate Metrics 계산 중...")
                climate_metrics = ClimateMetrics()
                metrics = climate_metrics.calculate_all_metrics(Xpred_ext, X_ext, dt=delta)
                
                print(f"    * Mean State Error: {metrics['mean_state_error_mean']:.6f}")
                print(f"    * Variance Ratio: {metrics['variance_ratio']:.6f}")
                print(f"    * KL Divergence: {metrics['kl_divergence']:.6f}")
                print(f"    * Extreme Event Freq (Pred/True): {metrics['extreme_event_freq_pred']:.4f}/{metrics['extreme_event_freq_true']:.4f}")
                
            except Exception as e:
                print(f"    * Climate Metrics 계산 중 오류: {e}")

        except Exception as e:
            print(f"시뮬레이션 중 오류 발생: {e}")
            continue

    # 평균 성능 계산
    if len(performance_scores) > 0:
        mean_performance = np.mean(performance_scores)
        std_performance = np.std(performance_scores)
        
        print(f"\n=== 최종 결과 ===")
        print(f"성공한 시도 수: {len(performance_scores)}/{num_trials}")
        print(f"개별 성능 점수: {[f'{score:.6f}' for score in performance_scores]}")
        print(f"평균 성능 점수: {mean_performance:.6f}")
        print(f"성능 점수 표준편차: {std_performance:.6f}")
        
        return performance_scores, mean_performance
    else:
        print("성공한 시도가 없습니다!")
        return [], None


def evaluate_extrapolation_performance_with_uncertainty(model, metadata, memory_length_TM, num_trials=5, t_end=10, t_start_plot=2, delta=0.005, num_mc_samples=100):
    """
    랜덤 초기 조건에 대해 불확실성을 포함한 extrapolation 성능을 평가하는 함수
    논문의 성능 측정 방법: np.linalg.norm(X_ext-Xpred_ext) / np.linalg.norm(X_ext)
    
    Args:
        model: 학습된 dAMZ 모델 (Monte Carlo Dropout 포함)
        metadata: Lorenz 96 시스템의 메타데이터
        num_trials: 평가할 랜덤 초기 조건의 수
        t_end: 적분 끝 시간
        t_start_plot: 시각화 시작 시간
        delta: 시간 간격
        num_mc_samples: Monte Carlo 샘플 수
    
    Returns:
        performance_scores: 각 시도별 성능 점수 리스트
        mean_performance: 평균 성능 점수
        uncertainty_scores: 각 시도별 평균 불확실성 점수 리스트
    """
    # t_end가 t_start_plot보다 큰지 확인하고 예외 처리
    if t_end <= t_start_plot:
        print(f"경고: t_end({t_end})가 t_start_plot({t_start_plot})보다 작거나 같습니다.")
        print(f"t_end를 {t_start_plot + 1.0}로 조정합니다.")
        t_end = t_start_plot + 1.0
    
    # 시스템 파라미터 추출
    K = metadata['K']
    J = metadata['J']
    F = metadata['F']
    h = metadata['h']
    b = metadata['b']
    c = metadata['c']
    dt = metadata['dt']
    
    performance_scores = []
    uncertainty_scores = []
    
    print(f"\n=== Extrapolation 성능 평가 (불확실성 포함, 랜덤 초기 조건 {num_trials}개) ===")
    
    for trial in range(num_trials):
        print(f"\n--- 시도 {trial + 1}/{num_trials} ---")
        
        # 랜덤 초기 조건 생성
        np.random.seed(42 + trial)  # 재현성을 위한 시드 설정
        
        # X 초기 조건: 논문과 유사한 범위로 설정
        X_init = np.random.uniform(-5, 5, K)
        
        # Y 초기 조건: 작은 값으로 설정
        Y_init = np.random.uniform(-0.1, 0.1, K * J)
        
        initial_condition = np.concatenate([X_init, Y_init])
        print(f"초기 조건 X: {X_init}")
        print(f"초기 조건 Y 범위: [{Y_init.min():.3f}, {Y_init.max():.3f}]")

        # 실제 시뮬레이션
        t_eval = np.arange(0, t_end, delta)  # t_end는 포함하지 않음
        t_span = (0, t_end)

        try:
            # Lorenz 96 시스템 적분
            sol = solve_ivp(
                fun=lambda t, y: lorenz96_system_equations(t, y, F, h, b, c, K, J),
                t_span=t_span,
                y0=initial_condition,
                t_eval=t_eval,
                method='RK45',
                rtol=1e-9,
                atol=1e-11
            )

            t_vals = sol.t
            X_true = sol.y[:K, :].T  # X 변수들만 추출

            # 시스템 발산 감지
            if np.any(np.abs(X_true) > 1000) or len(t_vals) < t_end / delta * 0.5:
                print("경고: 시스템이 발산했습니다. 이 시도는 건너뜁니다.")
                continue

            # 모델 예측 (불확실성 포함)
            n_M = int(memory_length_TM / dt)  # 메모리 길이 (일관성을 위해 동일하게 설정)

            X_pred_mean = []
            X_pred_std = []
            
            with torch.no_grad():
                for i in range(n_M + 1, len(X_true)):
                    # Z = (z_n^T, z_{n-1}^T, ..., z_{n-nM}^T)^T
                    Z_input = X_true[i-n_M-1:i].reshape(-1)  # [D]
                    Z_tensor = torch.FloatTensor(Z_input).unsqueeze(0)  # [1, D]

                    # Monte Carlo Dropout을 사용한 불확실성 추정
                    mean_pred, std_pred = model.predict_with_uncertainty(Z_tensor, num_samples=num_mc_samples)
                    X_pred_mean.append(mean_pred[0].numpy())
                    X_pred_std.append(std_pred[0].numpy())

            X_pred_mean = np.array(X_pred_mean)
            X_pred_std = np.array(X_pred_std)
            t_pred = t_vals[n_M + 1:]
            
            # extrapolation 구간 설정 (t_start_plot 이후)
            mask = (t_pred >= t_start_plot)
            X_ext = X_true[n_M + 1:][mask]  # 실제 extrapolation 데이터
            Xpred_ext = X_pred_mean[mask]  # 예측 extrapolation 데이터 (평균)
            Xpred_std_ext = X_pred_std[mask]  # 예측 불확실성

            if len(X_ext) == 0:
                print("경고: extrapolation 데이터가 없습니다!")
                continue

            # 논문의 성능 측정 방법 적용
            # np.linalg.norm(X_ext-Xpred_ext) / np.linalg.norm(X_ext)
            numerator = np.linalg.norm(X_ext - Xpred_ext)
            denominator = np.linalg.norm(X_ext)
            
            if denominator == 0:
                print("경고: 분모가 0입니다!")
                continue
                
            performance_score = numerator / denominator
            performance_scores.append(performance_score)
            
            # 평균 불확실성 계산
            mean_uncertainty = np.mean(Xpred_std_ext)
            uncertainty_scores.append(mean_uncertainty)
            
            print(f"Extrapolation 성능 점수: {performance_score:.6f}")
            print(f"  - 분자 (예측 오차 norm): {numerator:.6f}")
            print(f"  - 분모 (실제 데이터 norm): {denominator:.6f}")
            print(f"  - 평균 불확실성: {mean_uncertainty:.6f}")
            print(f"  - 데이터 포인트 수: {len(X_ext)}")
            
            # Climate Metrics 계산 추가 (불확실성 포함)
            try:
                print("  - Climate Metrics 계산 중...")
                climate_metrics = ClimateMetrics()
                metrics = climate_metrics.calculate_all_metrics(Xpred_ext, X_ext, dt=delta)
                
                print(f"    * Mean State Error: {metrics['mean_state_error_mean']:.6f}")
                print(f"    * Variance Ratio: {metrics['variance_ratio']:.6f}")
                print(f"    * KL Divergence: {metrics['kl_divergence']:.6f}")
                print(f"    * Extreme Event Freq (Pred/True): {metrics['extreme_event_freq_pred']:.4f}/{metrics['extreme_event_freq_true']:.4f}")
                
                # 불확실성과 관련된 추가 지표
                print(f"    * Prediction Std (평균): {np.mean(Xpred_std_ext):.6f}")
                
            except Exception as e:
                print(f"    * Climate Metrics 계산 중 오류: {e}")

        except Exception as e:
            print(f"시뮬레이션 중 오류 발생: {e}")
            continue

    # 평균 성능 계산
    if len(performance_scores) > 0:
        mean_performance = np.mean(performance_scores)
        std_performance = np.std(performance_scores)
        mean_uncertainty = np.mean(uncertainty_scores)
        
        print(f"\n=== 최종 결과 ===")
        print(f"성공한 시도 수: {len(performance_scores)}/{num_trials}")
        print(f"개별 성능 점수: {[f'{score:.6f}' for score in performance_scores]}")
        print(f"평균 성능 점수: {mean_performance:.6f}")
        print(f"성능 점수 표준편차: {std_performance:.6f}")
        print(f"평균 불확실성: {mean_uncertainty:.6f}")
        
        return performance_scores, mean_performance, uncertainty_scores
    else:
        print("성공한 시도가 없습니다!")
        return [], None, []




def evaluate_climate_metrics_from_states(u_true, u_pred, time_axis=None, save_path=None):
    """
    이미 계산된 true state와 forecasted state를 받아서 기후 예측 지표들을 계산
    
    Args:
        u_true (np.ndarray): 실제 상태 [time_steps, variables] 또는 [time_steps]
        u_pred (np.ndarray): 예측 상태 [time_steps, variables] 또는 [time_steps]
        time_axis (np.ndarray): 시간 축 (None인 경우 인덱스 사용)
        save_path (str): 저장할 파일 경로 (None인 경우 저장하지 않음)
        
    Returns:
        dict: 모든 평가 지표를 포함한 딕셔너리
    """
    print("\n[+] Climate Metrics 계산 시작...")
    
    # ClimateMetrics 클래스 인스턴스 생성
    climate_metrics = ClimateMetrics()
    
    # 모든 지표 계산 (dt는 기본값 0.005 사용)
    metrics = climate_metrics.calculate_all_metrics(u_pred, u_true, dt=0.005)
    
    # 결과 출력
    climate_metrics.print_metrics_summary(metrics)
    
    # 시각화
    climate_metrics.plot_metrics_over_time(metrics, time_axis, save_path)
    
    return metrics


def evaluate_climate_metrics_with_uncertainty(u_true, u_pred_samples, time_axis=None, save_path=None):
    """
    불확실성을 포함한 예측 상태들을 받아서 기후 예측 지표들을 계산
    
    Args:
        u_true (np.ndarray): 실제 상태 [time_steps, variables] 또는 [time_steps]
        u_pred_samples (list): Monte Carlo 샘플들의 리스트 [num_samples, time_steps, variables]
        time_axis (np.ndarray): 시간 축 (None인 경우 인덱스 사용)
        save_path (str): 저장할 파일 경로 (None인 경우 저장하지 않음)
        
    Returns:
        dict: 모든 평가 지표를 포함한 딕셔너리 (불확실성 포함)
    """
    print("\n[+] Climate Metrics 계산 (불확실성 포함)...")
    
    # ClimateMetrics 클래스 인스턴스 생성
    climate_metrics = ClimateMetrics()
    
    # 예측값 평균 계산
    u_pred_mean = np.mean(u_pred_samples, axis=0)
    
    # 모든 지표 계산 (평균 예측값 기준, dt는 기본값 0.005 사용)
    metrics = climate_metrics.calculate_all_metrics(u_pred_mean, u_true, dt=0.005)
    
    # 불확실성 계산을 위한 추가 지표들
    metrics['prediction_std'] = np.std(u_pred_samples, axis=0)
    metrics['prediction_std_mean'] = np.mean(metrics['prediction_std'])
    
    # 각 Monte Carlo 샘플에 대한 지표들 계산
    all_metrics_samples = []
    for i, u_pred_sample in enumerate(u_pred_samples):
        sample_metrics = climate_metrics.calculate_all_metrics(u_pred_sample, u_true, dt=0.005)
        all_metrics_samples.append(sample_metrics)
    
    # 지표들의 표준편차 계산 (불확실성 측정)
    metrics['mean_state_error_std'] = np.std([m['mean_state_error_mean'] for m in all_metrics_samples])
    metrics['variance_ratio_std'] = np.std([m['variance_ratio'] for m in all_metrics_samples])
    metrics['kl_divergence_std'] = np.std([m['kl_divergence'] for m in all_metrics_samples])
    metrics['extreme_event_freq_pred_std'] = np.std([m['extreme_event_freq_pred'] for m in all_metrics_samples])
    
    # 결과 출력
    print("\n" + "="*70)
    print("           CLIMATE METRICS WITH UNCERTAINTY")
    print("="*70)
    print(f"Mean State Error (평균): {metrics['mean_state_error_mean']:.6f} ± {metrics['mean_state_error_std']:.6f}")
    print(f"Variance Ratio (분산 비율): {metrics['variance_ratio']:.6f} ± {metrics['variance_ratio_std']:.6f}")
    print(f"KL Divergence (쿨백-라이블러 발산): {metrics['kl_divergence']:.6f} ± {metrics['kl_divergence_std']:.6f}")
    print(f"Extreme Event Frequency - Prediction: {metrics['extreme_event_freq_pred']:.4f} ± {metrics['extreme_event_freq_pred_std']:.4f}")
    print(f"Extreme Event Frequency - True: {metrics['extreme_event_freq_true']:.4f}")
    print(f"Prediction Standard Deviation (평균): {metrics['prediction_std_mean']:.6f}")
    print("="*70)
    
    # 시각화
    climate_metrics.plot_metrics_over_time(metrics, time_axis, save_path)
    
    return metrics
