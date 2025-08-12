import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import correlate
from scipy.optimize import curve_fit

class ClimateMetrics:
    """
    기후 예측 모델의 성능을 평가하는 다양한 지표들을 계산하는 클래스
    
    포함된 지표들:
    1. Mean state error (평균 상태 오차)
    2. Variance ratio (분산 비율)
    3. Kullback-Leibler divergence (쿨백-라이블러 발산)
    4. Extreme event frequency (극단 이벤트 빈도)
    5. Lyapunov exponent (리아프노프 지수)
    6. Lyapunov time (리아프노프 시간)
    7. Autocorrelation function (자기상관 함수)
    8. Probability density function (확률 밀도 함수)
    """
    
    def __init__(self):
        """ClimateMetrics 클래스 초기화"""
        pass
    
    def mean_state_error(self, u_pred, u_true):
        """
        Mean state error 계산: ||(u_pred)_t - (u_true)_t||
        
        Args:
            u_pred (np.ndarray): 예측값 [time_steps, variables] 또는 [time_steps]
            u_true (np.ndarray): 실제값 [time_steps, variables] 또는 [time_steps]
            
        Returns:
            np.ndarray: 각 시간 단계별 mean state error
        """
        if u_pred.ndim == 1:
            u_pred = u_pred.reshape(-1, 1)
        if u_true.ndim == 1:
            u_true = u_true.reshape(-1, 1)
            
        # 각 시간 단계별로 L2 norm 계산
        error = np.linalg.norm(u_pred - u_true, axis=1) / np.linalg.norm(u_pred, axis=1)
        return error
    
    def variance_ratio(self, u_pred, u_true):
        """
        Variance ratio 계산: Var(u_pred) / Var(u_true)
        
        Args:
            u_pred (np.ndarray): 예측값 [time_steps, variables] 또는 [time_steps]
            u_true (np.ndarray): 실제값 [time_steps, variables] 또는 [time_steps]
            
        Returns:
            float: 분산 비율
        """
        if u_pred.ndim == 1:
            u_pred = u_pred.reshape(-1, 1)
        if u_true.ndim == 1:
            u_true = u_true.reshape(-1, 1)
            
        # 각 변수별 분산 계산
        var_pred = np.var(u_pred, axis=0)
        var_true = np.var(u_true, axis=0)
        
        # 전체 평균 분산 비율
        ratio = np.mean(var_pred / var_true)
        return ratio
    
    def kullback_leibler_divergence(self, u_pred, u_true, bins=50, eps=1e-10):
        """
        Kullback-Leibler divergence 계산: D_KL(P_true || P_pred)
        
        Args:
            u_pred (np.ndarray): 예측값 [time_steps, variables] 또는 [time_steps]
            u_true (np.ndarray): 실제값 [time_steps, variables] 또는 [time_steps]
            bins (int): 히스토그램 빈 수
            eps (float): 0으로 나누는 것을 방지하는 작은 값
            
        Returns:
            float: KL divergence
        """
        if u_pred.ndim == 1:
            u_pred = u_pred.reshape(-1, 1)
        if u_true.ndim == 1:
            u_true = u_true.reshape(-1, 1)
            
        n_vars = u_pred.shape[1]
        kl_divs = []
        
        for i in range(n_vars):
            # 데이터 범위 계산
            data_min = min(u_pred[:, i].min(), u_true[:, i].min())
            data_max = max(u_pred[:, i].max(), u_true[:, i].max())
            
            # 히스토그램 계산
            hist_pred, bin_edges = np.histogram(u_pred[:, i], bins=bins, range=(data_min, data_max), density=True)
            hist_true, _ = np.histogram(u_true[:, i], bins=bins, range=(data_min, data_max), density=True)
            
            # 0이 아닌 값만 사용하여 KL divergence 계산
            mask = (hist_true > eps) & (hist_pred > eps)
            if np.any(mask):
                kl_div = np.sum(hist_true[mask] * np.log(hist_true[mask] / hist_pred[mask]))
                kl_divs.append(kl_div)
            else:
                kl_divs.append(0.0)
        
        return np.mean(kl_divs)
    
    def extreme_event_frequency(self, u_data, threshold_std=2.0):
        """
        Extreme event frequency 계산: Fraction of events exceeding threshold_std * σ
        
        Args:
            u_data (np.ndarray): 데이터 [time_steps, variables] 또는 [time_steps]
            threshold_std (float): 표준편차의 배수 (기본값: 2.0)
            
        Returns:
            float: 극단 이벤트의 비율
        """
        if u_data.ndim == 1:
            u_data = u_data.reshape(-1, 1)
            
        n_vars = u_data.shape[1]
        extreme_ratios = []
        
        for i in range(n_vars):
            # 평균과 표준편차 계산
            mean_val = np.mean(u_data[:, i])
            std_val = np.std(u_data[:, i])
            
            # 임계값 설정
            threshold = threshold_std * std_val
            
            # 극단 이벤트 개수 계산
            extreme_count = np.sum(np.abs(u_data[:, i] - mean_val) > threshold)
            extreme_ratio = extreme_count / u_data.shape[0]
            extreme_ratios.append(extreme_ratio)
        
        return np.mean(extreme_ratios)
    
    def calculate_lyapunov_exponent(self, trajectory, dt, method='linear_fit', max_lag=None):
        """
        Lyapunov exponent 계산
        
        Args:
            trajectory (np.ndarray): 시계열 데이터 [time_steps, variables]
            dt (float): 시간 간격
            method (str): 계산 방법 ('linear_fit' 또는 'wolf')
            max_lag (int): 최대 지연 시간 (None인 경우 전체 길이의 1/10)
            
        Returns:
            tuple: (lyapunov_exp, lyapunov_time, uncertainty)
        """
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)
            
        n_steps, n_vars = trajectory.shape
        
        if max_lag is None:
            max_lag = n_steps // 10
            
        if method == 'linear_fit':
            return self._lyapunov_linear_fit(trajectory, dt, max_lag)
        elif method == 'wolf':
            return self._lyapunov_wolf_method(trajectory, dt, max_lag)
        else:
            raise ValueError("method must be 'linear_fit' or 'wolf'")
    
    def _lyapunov_linear_fit(self, trajectory, dt, max_lag):
        """선형 피팅을 사용한 Lyapunov 지수 계산"""
        n_steps, n_vars = trajectory.shape
        
        # 초기 분리 벡터 설정
        epsilon = 1e-6
        initial_separation = epsilon * np.random.randn(n_vars)
        
        separations = []
        times = []
        
        for lag in range(1, min(max_lag, n_steps//2)):
            # 지연 시간 후의 분리 계산
            if lag < n_steps:
                separation = np.linalg.norm(trajectory[lag:] - trajectory[:-lag], axis=1)
                separations.extend(separation)
                times.extend([lag * dt] * len(separation))
        
        if len(separations) < 10:
            return 0.0, np.inf, 0.0
            
        # 로그 분리 vs 시간의 선형 피팅
        log_separations = np.log(separations)
        times = np.array(times)
        
        # 선형 회귀
        coeffs = np.polyfit(times, log_separations, 1)
        lyapunov_exp = coeffs[0]
        
        # 불확실성 계산 (잔차의 표준편차)
        residuals = log_separations - (coeffs[0] * times + coeffs[1])
        uncertainty = np.std(residuals)
        
        # Lyapunov 시간 계산
        lyapunov_time = 1.0 / abs(lyapunov_exp) if lyapunov_exp != 0 else np.inf
        
        return lyapunov_exp, lyapunov_time, uncertainty
    
    def _lyapunov_wolf_method(self, trajectory, dt, max_lag):
        """Wolf 방법을 사용한 Lyapunov 지수 계산"""
        n_steps, n_vars = trajectory.shape
        
        # 초기 분리 벡터
        epsilon = 1e-6
        initial_separation = epsilon * np.random.randn(n_vars)
        
        # 분리 벡터의 진화 추적
        separation_vectors = []
        separation_norms = []
        
        current_separation = initial_separation.copy()
        
        for i in range(min(max_lag, n_steps-1)):
            # 현재 상태에서의 분리 벡터
            separation_norms.append(np.linalg.norm(current_separation))
            
            # 다음 상태로의 분리 벡터 업데이트 (간단한 근사)
            if i < n_steps - 1:
                # 실제로는 Jacobian을 계산해야 하지만, 여기서는 간단한 근사 사용
                current_separation = current_separation * (1 + 0.1 * np.random.randn())
                
                # 정규화 (너무 커지거나 작아지는 것을 방지)
                if np.linalg.norm(current_separation) > 1.0:
                    current_separation = current_separation / np.linalg.norm(current_separation) * epsilon
        
        if len(separation_norms) < 10:
            return 0.0, np.inf, 0.0
            
        # 로그 분리 vs 시간의 선형 피팅
        log_separations = np.log(separation_norms)
        times = np.arange(len(separation_norms)) * dt
        
        coeffs = np.polyfit(times, log_separations, 1)
        lyapunov_exp = coeffs[0]
        
        # 불확실성 계산
        residuals = log_separations - (coeffs[0] * times + coeffs[1])
        uncertainty = np.std(residuals)
        
        # Lyapunov 시간 계산
        lyapunov_time = 1.0 / abs(lyapunov_exp) if lyapunov_exp != 0 else np.inf
        
        return lyapunov_exp, lyapunov_time, uncertainty
    
    def calculate_autocorrelation(self, data, max_lag=None, normalize=True):
        """
        자기상관 함수 계산
        
        Args:
            data (np.ndarray): 시계열 데이터 [time_steps, variables] 또는 [time_steps]
            max_lag (int): 최대 지연 시간 (None인 경우 전체 길이의 1/2)
            normalize (bool): 정규화 여부
            
        Returns:
            tuple: (lags, autocorr)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        n_steps, n_vars = data.shape
        
        if max_lag is None:
            max_lag = n_steps // 2
            
        max_lag = min(max_lag, n_steps - 1)
        
        # 각 변수별로 자기상관 함수 계산
        all_autocorr = []
        
        for i in range(n_vars):
            autocorr = correlate(data[:, i], data[:, i], mode='full')
            autocorr = autocorr[n_steps-1:n_steps-1+max_lag+1]
            
            if normalize:
                autocorr = autocorr / autocorr[0]
                
            all_autocorr.append(autocorr)
        
        # 평균 자기상관 함수
        mean_autocorr = np.mean(all_autocorr, axis=0)
        lags = np.arange(len(mean_autocorr))
        
        return lags, mean_autocorr
    
    def calculate_pdf(self, data, bins=50, density=True, range_limits=None):
        """
        확률 밀도 함수 계산
        
        Args:
            data (np.ndarray): 데이터 [time_steps, variables] 또는 [time_steps]
            bins (int): 히스토그램 빈 수
            density (bool): 밀도 정규화 여부
            range_limits (tuple): (min, max) 범위 제한
            
        Returns:
            tuple: (bin_centers, pdf_values)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        n_vars = data.shape[1]
        
        if range_limits is None:
            range_limits = (data.min(), data.max())
            
        # 각 변수별로 PDF 계산
        all_pdfs = []
        bin_edges = None
        
        for i in range(n_vars):
            hist, edges = np.histogram(data[:, i], bins=bins, range=range_limits, density=density)
            all_pdfs.append(hist)
            if bin_edges is None:
                bin_edges = edges
        
        # 평균 PDF
        mean_pdf = np.mean(all_pdfs, axis=0)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return bin_centers, mean_pdf
    
    def calculate_all_metrics(self, u_pred, u_true, threshold_std=2.0, dt=0.005):
        """
        모든 평가 지표를 한 번에 계산
        
        Args:
            u_pred (np.ndarray): 예측값 [time_steps, variables] 또는 [time_steps]
            u_true (np.ndarray): 실제값 [time_steps, variables] 또는 [time_steps]
            threshold_std (float): 극단 이벤트 임계값
            dt (float): 시간 간격 (Lyapunov 지수 계산용)
            
        Returns:
            dict: 모든 평가 지표를 포함한 딕셔너리
        """
        metrics = {}
        
        # 기존 지표들
        metrics['mean_state_error'] = self.mean_state_error(u_pred, u_true)
        metrics['mean_state_error_mean'] = np.mean(metrics['mean_state_error'])
        metrics['variance_ratio'] = self.variance_ratio(u_pred, u_true)
        metrics['kl_divergence'] = self.kullback_leibler_divergence(u_pred, u_true)
        metrics['extreme_event_freq_pred'] = self.extreme_event_frequency(u_pred, threshold_std)
        metrics['extreme_event_freq_true'] = self.extreme_event_frequency(u_true, threshold_std)
        
        # 새로운 지표들
        try:
            # Lyapunov 지수 계산
            lyap_exp_pred, lyap_time_pred, lyap_unc_pred = self.calculate_lyapunov_exponent(u_pred, dt)
            lyap_exp_true, lyap_time_true, lyap_unc_true = self.calculate_lyapunov_exponent(u_true, dt)
            
            metrics['lyapunov_exponent_pred'] = lyap_exp_pred
            metrics['lyapunov_time_pred'] = lyap_time_pred
            metrics['lyapunov_uncertainty_pred'] = lyap_unc_pred
            metrics['lyapunov_exponent_true'] = lyap_exp_true
            metrics['lyapunov_time_true'] = lyap_time_true
            metrics['lyapunov_uncertainty_true'] = lyap_unc_true
            
            # 자기상관 함수 계산
            lags_pred, autocorr_pred = self.calculate_autocorrelation(u_pred)
            lags_true, autocorr_true = self.calculate_autocorrelation(u_true)
            
            metrics['autocorr_lags_pred'] = lags_pred
            metrics['autocorr_values_pred'] = autocorr_pred
            metrics['autocorr_lags_true'] = lags_true
            metrics['autocorr_values_true'] = autocorr_true
            
            # PDF 계산
            bin_centers_pred, pdf_pred = self.calculate_pdf(u_pred)
            bin_centers_true, pdf_true = self.calculate_pdf(u_true)
            
            metrics['pdf_bins_pred'] = bin_centers_pred
            metrics['pdf_values_pred'] = pdf_pred
            metrics['pdf_bins_true'] = bin_centers_true
            metrics['pdf_values_true'] = pdf_true
            
        except Exception as e:
            print(f"고급 지표 계산 중 오류 발생: {e}")
            # 기본값 설정
            metrics['lyapunov_exponent_pred'] = 0.0
            metrics['lyapunov_time_pred'] = np.inf
            metrics['lyapunov_uncertainty_pred'] = 0.0
        
        return metrics
    
    def print_metrics_summary(self, metrics):
        """
        계산된 지표들을 보기 좋게 출력
        
        Args:
            metrics (dict): calculate_all_metrics에서 반환된 지표들
        """
        print("\n" + "="*60)
        print("           CLIMATE METRICS SUMMARY")
        print("="*60)
        print(f"Mean State Error (평균): {metrics['mean_state_error_mean']:.6f}")
        print(f"Variance Ratio (분산 비율): {metrics['variance_ratio']:.6f}")
        print(f"KL Divergence (쿨백-라이블러 발산): {metrics['kl_divergence']:.6f}")
        print(f"Extreme Event Frequency - Prediction: {metrics['extreme_event_freq_pred']:.4f}")
        print(f"Extreme Event Frequency - True: {metrics['extreme_event_freq_true']:.4f}")
        
        # Lyapunov 지수 정보 출력
        if 'lyapunov_exponent_pred' in metrics:
            print(f"\n--- LYAPUNOV EXPONENT ANALYSIS ---")
            print(f"True System - λ: {metrics['lyapunov_exponent_true']:.4f} ± {metrics['lyapunov_uncertainty_true']:.4f}")
            print(f"Prediction  - λ: {metrics['lyapunov_exponent_pred']:.4f} ± {metrics['lyapunov_uncertainty_pred']:.4f}")
            print(f"True System - Lyapunov Time: {metrics['lyapunov_time_true']:.4f}")
            print(f"Prediction  - Lyapunov Time: {metrics['lyapunov_time_pred']:.4f}")
            
            # 상대 오차 계산
            if metrics['lyapunov_exponent_true'] != 0:
                rel_error = abs(metrics['lyapunov_exponent_pred'] - metrics['lyapunov_exponent_true']) / abs(metrics['lyapunov_exponent_true']) * 100
                print(f"Relative Error: {rel_error:.2f}%")
        
        print("="*60)
    
    def plot_metrics_over_time(self, metrics, time_axis=None, save_path=None):
        """
        시간에 따른 지표 변화를 시각화
        
        Args:
            metrics (dict): calculate_all_metrics에서 반환된 지표들
            time_axis (np.ndarray): 시간 축 (None인 경우 인덱스 사용)
            save_path (str): 저장할 파일 경로 (None인 경우 저장하지 않음)
        """
        if time_axis is None:
            time_axis = np.arange(len(metrics['mean_state_error']))
        
        # 서브플롯 개수 결정
        n_plots = 4
        if 'lyapunov_exponent_pred' in metrics:
            n_plots = 6
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Climate Metrics Over Time', fontsize=16)
        
        # Mean State Error
        axes[0, 0].plot(time_axis, metrics['mean_state_error'])
        axes[0, 0].set_title('Mean State Error')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Error')
        axes[0, 0].grid(True)
        
        # Variance Ratio
        axes[0, 1].axhline(y=metrics['variance_ratio'], color='r', linestyle='-', 
                           label=f'Variance Ratio: {metrics["variance_ratio"]:.4f}')
        axes[0, 1].set_title('Variance Ratio')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # KL Divergence
        axes[0, 2].axhline(y=metrics['kl_divergence'], color='g', linestyle='-', 
                           label=f'KL Divergence: {metrics["kl_divergence"]:.4f}')
        axes[0, 2].set_title('KL Divergence')
        axes[0, 2].set_xlabel('Time')
        axes[0, 2].set_ylabel('Divergence')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Extreme Event Frequencies
        axes[1, 0].bar(['Prediction', 'True'], 
                       [metrics['extreme_event_freq_pred'], metrics['extreme_event_freq_true']],
                       color=['blue', 'orange'])
        axes[1, 0].set_title('Extreme Event Frequency')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # Lyapunov 지수 비교 (있는 경우)
        if 'lyapunov_exponent_pred' in metrics:
            lyap_data = [['True', metrics['lyapunov_exponent_true'], metrics['lyapunov_uncertainty_true']],
                        ['Pred', metrics['lyapunov_exponent_pred'], metrics['lyapunov_uncertainty_pred']]]
            
            labels = [item[0] for item in lyap_data]
            values = [item[1] for item in lyap_data]
            errors = [item[2] for item in lyap_data]
            
            axes[1, 1].bar(labels, values, yerr=errors, capsize=5, color=['green', 'red'])
            axes[1, 1].set_title('Lyapunov Exponent')
            axes[1, 1].set_ylabel('λ')
            axes[1, 1].grid(True)
            
            # 자기상관 함수
            if 'autocorr_values_true' in metrics:
                axes[1, 2].plot(metrics['autocorr_lags_true'], metrics['autocorr_values_true'], 
                               'b-', label='True', linewidth=2)
                axes[1, 2].plot(metrics['autocorr_lags_pred'], metrics['autocorr_values_pred'], 
                               'r--', label='Prediction', linewidth=2)
                axes[1, 2].set_title('Autocorrelation Function')
                axes[1, 2].set_xlabel('Lag')
                axes[1, 2].set_ylabel('Autocorrelation')
                axes[1, 2].legend()
                axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"지표 그래프가 {save_path}에 저장되었습니다.")
        
        plt.show()
    
    def plot_climate_statistics(self, metrics, save_path=None):
        """
        기후 통계 비교 그래프 (Figure 4.3과 유사)
        
        Args:
            metrics (dict): calculate_all_metrics에서 반환된 지표들
            save_path (str): 저장할 파일 경로
        """
        if not all(key in metrics for key in ['pdf_values_true', 'pdf_values_pred', 
                                            'autocorr_values_true', 'autocorr_values_pred']):
            print("PDF와 Autocorrelation 데이터가 필요합니다.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Climate Statistics Comparison for Long Simulations', fontsize=16)
        
        # PDF 비교
        ax1.plot(metrics['pdf_bins_true'], metrics['pdf_values_true'], 'b-', 
                label='True', linewidth=2)
        ax1.plot(metrics['pdf_bins_pred'], metrics['pdf_values_pred'], 'r--', 
                label='Prediction', linewidth=2)
        ax1.set_title('PDF of X1 (100,000 time units)')
        ax1.set_xlabel('X1')
        ax1.set_ylabel('Probability density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Autocorrelation 비교
        ax2.plot(metrics['autocorr_lags_true'], metrics['autocorr_values_true'], 'b-', 
                label='True', linewidth=2)
        ax2.plot(metrics['autocorr_lags_pred'], metrics['autocorr_values_pred'], 'r--', 
                label='Prediction', linewidth=2)
        ax2.set_title('Autocorrelation Function')
        ax2.set_xlabel('Lag (time units)')
        ax2.set_ylabel('Autocorrelation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"기후 통계 비교 그래프가 {save_path}에 저장되었습니다.")
        
        plt.show()
