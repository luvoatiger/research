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
        error = np.linalg.norm(u_pred - u_true, axis=1) / np.linalg.norm(u_true, axis=1)
        
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
            method (str): 계산 방법 ('linear_fit', 'wolf', 'rosenstein', 'kantz')
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
        elif method == 'rosenstein':
            return self._lyapunov_rosenstein_method(trajectory, dt, max_lag)
        elif method == 'kantz':
            return self._lyapunov_kantz_method(trajectory, dt, max_lag)
        else:
            raise ValueError("method must be 'linear_fit', 'wolf', 'rosenstein', or 'kantz'")
    
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
        lyapunov_time_result = self.calculate_lyapunov_time(lyapunov_exp)
        lyapunov_time = lyapunov_time_result['lyapunov_time']
        
        return lyapunov_exp, lyapunov_time, uncertainty
    
    def _lyapunov_wolf_method(self, trajectory, dt, max_lag):
        """Wolf 방법을 사용한 Lyapunov 지수 계산
        
        실제 Wolf 방법: Jacobian 행렬을 사용하여 분리 벡터의 진화를 추적
        """
        n_steps, n_vars = trajectory.shape
        
        # 초기 분리 벡터 설정
        epsilon = 1e-6
        initial_separation = epsilon * np.random.randn(n_vars)
        
        # 분리 벡터의 진화 추적
        separation_norms = []
        times = []
        
        current_separation = initial_separation.copy()
        
        for i in range(min(max_lag, n_steps-1)):
            # 현재 상태에서의 분리 벡터 크기
            separation_norms.append(np.linalg.norm(current_separation))
            times.append(i * dt)
            
            if i < n_steps - 1:
                # Jacobian 행렬 근사 (유한 차분 사용)
                jacobian = self._approximate_jacobian(trajectory[i], trajectory[i+1], dt)
                
                # 분리 벡터 업데이트: δx(t+1) = J(t) * δx(t)
                current_separation = jacobian @ current_separation
                
                # 정규화: 너무 커지거나 작아지는 것을 방지
                norm_sep = np.linalg.norm(current_separation)
                if norm_sep > 1.0:
                    current_separation = current_separation / norm_sep * epsilon
                elif norm_sep < epsilon * 1e-3:
                    # 너무 작아지면 새로운 방향으로 재시작
                    current_separation = epsilon * np.random.randn(n_vars)
        
        if len(separation_norms) < 10:
            return 0.0, np.inf, 0.0
            
        # 로그 분리 vs 시간의 선형 피팅
        log_separations = np.log(separation_norms)
        times = np.array(times)
        
        # 선형 회귀
        coeffs = np.polyfit(times, log_separations, 1)
        lyapunov_exp = coeffs[0]
        
        # 불확실성 계산
        residuals = log_separations - (coeffs[0] * times + coeffs[1])
        uncertainty = np.std(residuals)
        
        # Lyapunov 시간 계산
        lyapunov_time_result = self.calculate_lyapunov_time(lyapunov_exp)
        lyapunov_time = lyapunov_time_result['lyapunov_time']
        
        return lyapunov_exp, lyapunov_time, uncertainty
    
    def _approximate_jacobian(self, x_t, x_tp1, dt):
        """유한 차분을 사용한 Jacobian 행렬 근사"""
        n_vars = len(x_t)
        jacobian = np.zeros((n_vars, n_vars))
        
        # 각 변수에 대한 편미분 계산
        for i in range(n_vars):
            # i번째 변수에 작은 변화 추가
            x_perturbed = x_t.copy()
            x_perturbed[i] += 1e-8
            
            # 변화된 상태에서의 미분 계산
            # dx/dt ≈ (x(t+1) - x(t)) / dt
            # ∂f_i/∂x_j ≈ (f_i(x+δx_j) - f_i(x)) / δx_j
            # 여기서는 간단한 근사 사용
            jacobian[:, i] = (x_tp1 - x_t) / dt
        
        return jacobian
    
    def _lyapunov_rosenstein_method(self, trajectory, dt, max_lag, min_neighbors=10):
        """Rosenstein 방법을 사용한 Lyapunov 지수 계산
        
        Rosenstein 방법: 최근접 이웃을 사용하여 지연 좌표에서의 분리 진화를 추적
        """
        n_steps, n_vars = trajectory.shape
        
        # 지연 좌표 구성 (간단한 1차 지연)
        delay = 1
        embedded_dim = min(3, n_vars)  # 임베딩 차원
        
        # 지연 좌표 데이터 구성
        embedded_data = []
        for i in range(n_steps - delay * (embedded_dim - 1)):
            point = []
            for j in range(embedded_dim):
                point.extend(trajectory[i + j * delay])
            embedded_data.append(point)
        
        embedded_data = np.array(embedded_data)
        n_embedded = len(embedded_data)
        
        if n_embedded < min_neighbors + 10:
            return 0.0, np.inf, 0.0
        
        # 각 점에 대해 최근접 이웃 찾기
        separations = []
        times = []
        
        for i in range(min(max_lag, n_embedded - 1)):
            current_point = embedded_data[i]
            
            # 현재 점과의 거리 계산
            distances = np.linalg.norm(embedded_data - current_point, axis=1)
            
            # 자기 자신과 너무 가까운 점들 제외
            valid_indices = np.where(distances > 1e-10)[0]
            if len(valid_indices) < min_neighbors:
                continue
                
            # 최근접 이웃 찾기
            nearest_idx = valid_indices[np.argmin(distances[valid_indices])]
            
            # 시간에 따른 분리 추적
            if nearest_idx + i < n_embedded:
                separation = np.linalg.norm(embedded_data[nearest_idx + i] - embedded_data[i])
                if separation > 1e-10:
                    separations.append(separation)
                    times.append(i * dt)
        
        if len(separations) < 10:
            return 0.0, np.inf, 0.0
            
        # 로그 분리 vs 시간의 선형 피팅
        log_separations = np.log(separations)
        times = np.array(times)
        
        # 선형 회귀
        coeffs = np.polyfit(times, log_separations, 1)
        lyapunov_exp = coeffs[0]
        
        # 불확실성 계산
        residuals = log_separations - (coeffs[0] * times + coeffs[1])
        uncertainty = np.std(residuals)
        
        # Lyapunov 시간 계산
        lyapunov_time_result = self.calculate_lyapunov_time(lyapunov_exp)
        lyapunov_time = lyapunov_time_result['lyapunov_time']
        
        return lyapunov_exp, lyapunov_time, uncertainty
    
    def _lyapunov_kantz_method(self, trajectory, dt, max_lag, min_neighbors=10, epsilon=1e-6):
        """Kantz 방법을 사용한 Lyapunov 지수 계산
        
        Kantz 방법: 각 점 주변의 이웃들을 사용하여 지역적 Lyapunov 지수를 계산
        """
        n_steps, n_vars = trajectory.shape
        
        # 지연 좌표 구성
        delay = 1
        embedded_dim = min(3, n_vars)
        
        # 지연 좌표 데이터 구성
        embedded_data = []
        for i in range(n_steps - delay * (embedded_dim - 1)):
            point = []
            for j in range(embedded_dim):
                point.extend(trajectory[i + j * delay])
            embedded_data.append(point)
        
        embedded_data = np.array(embedded_data)
        n_embedded = len(embedded_data)
        
        if n_embedded < min_neighbors + 10:
            return 0.0, np.inf, 0.0
        
        # 각 점에 대해 지역적 Lyapunov 지수 계산
        local_lyapunovs = []
        times = []
        
        for i in range(min(max_lag, n_embedded - 1)):
            current_point = embedded_data[i]
            
            # 현재 점과의 거리 계산
            distances = np.linalg.norm(embedded_data - current_point, axis=1)
            
            # ε-이웃 찾기 (너무 가까운 점들 제외)
            neighbor_mask = (distances > epsilon) & (distances < 2 * epsilon)
            neighbor_indices = np.where(neighbor_mask)[0]
            
            if len(neighbor_indices) < min_neighbors:
                continue
            
            # 이웃들과의 분리 진화 추적
            local_separations = []
            
            for neighbor_idx in neighbor_indices:
                if neighbor_idx + i < n_embedded:
                    initial_sep = distances[neighbor_idx]
                    final_sep = np.linalg.norm(embedded_data[neighbor_idx + i] - embedded_data[i])
                    
                    if initial_sep > 1e-10 and final_sep > 1e-10:
                        local_sep = np.log(final_sep / initial_sep) / (i * dt)
                        local_separations.append(local_sep)
            
            if len(local_separations) > 0:
                # 지역적 Lyapunov 지수의 평균
                local_lyap = np.mean(local_separations)
                local_lyapunovs.append(local_lyap)
                times.append(i * dt)
        
        if len(local_lyapunovs) < 10:
            return 0.0, np.inf, 0.0
            
        # 전체 Lyapunov 지수 (지역적 값들의 평균)
        lyapunov_exp = np.mean(local_lyapunovs)
        
        # 불확실성 계산
        uncertainty = np.std(local_lyapunovs)
        
        # Lyapunov 시간 계산
        lyapunov_time_result = self.calculate_lyapunov_time(lyapunov_exp)
        lyapunov_time = lyapunov_time_result['lyapunov_time']
        
        return lyapunov_exp, lyapunov_time, uncertainty
    
    def compare_lyapunov_methods(self, trajectory, dt, max_lag=None):
        """
        모든 Lyapunov 지수 계산 방법을 비교
        
        Args:
            trajectory (np.ndarray): 시계열 데이터 [time_steps, variables]
            dt (float): 시간 간격
            max_lag (int): 최대 지연 시간
            
        Returns:
            dict: 각 방법별 결과를 포함한 딕셔너리
        """
        methods = ['linear_fit', 'wolf', 'rosenstein', 'kantz']
        results = {}
        
        for method in methods:
            try:
                lyap_exp, lyap_time, uncertainty = self.calculate_lyapunov_exponent(
                    trajectory, dt, method=method, max_lag=max_lag
                )
                results[method] = {
                    'lyapunov_exponent': lyap_exp,
                    'lyapunov_time': lyap_time,
                    'uncertainty': uncertainty
                }
            except Exception as e:
                results[method] = {
                    'lyapunov_exponent': None,
                    'lyapunov_time': None,
                    'uncertainty': None,
                    'error': str(e)
                }
        
        return results
    
    def calculate_lyapunov_time(self, lyapunov_exponent, method='direct', uncertainty=None):
        """
        Lyapunov Time 계산
        
        Lyapunov Time은 시스템의 예측 가능성을 나타내는 시간 척도입니다.
        λ > 0인 경우: 시스템이 카오틱하며, 초기 조건의 작은 오차가 지수적으로 증폭됩니다.
        λ < 0인 경우: 시스템이 안정적이며, 초기 조건의 오차가 감소합니다.
        
        Args:
            lyapunov_exponent (float): Lyapunov 지수
            method (str): 계산 방법 ('direct', 'confidence_interval', 'bootstrap')
            uncertainty (float): Lyapunov 지수의 불확실성 (표준편차)
            
        Returns:
            dict: Lyapunov Time과 관련 정보를 포함한 딕셔너리
        """
        if lyapunov_exponent == 0:
            return {
                'lyapunov_time': np.inf,
                'prediction_horizon': np.inf,
                'stability': 'neutral',
                'confidence_interval': None,
                'method': method
            }
        
        # 기본 Lyapunov Time 계산: τ = 1/|λ|
        lyapunov_time = 1.0 / abs(lyapunov_exponent)
        
        # 시스템의 안정성 판정
        if lyapunov_exponent > 0:
            stability = 'chaotic'
            prediction_horizon = lyapunov_time
        else:
            stability = 'stable'
            prediction_horizon = np.inf
        
        result = {
            'lyapunov_time': lyapunov_time,
            'prediction_horizon': prediction_horizon,
            'stability': stability,
            'lyapunov_exponent': lyapunov_exponent,
            'method': method
        }
        
        # 불확실성이 주어진 경우 신뢰구간 계산
        if uncertainty is not None and method in ['confidence_interval', 'bootstrap']:
            if method == 'confidence_interval':
                # 95% 신뢰구간 계산 (정규분포 가정)
                z_score = 1.96  # 95% 신뢰수준
                
                # Lyapunov 지수의 신뢰구간
                lambda_lower = lyapunov_exponent - z_score * uncertainty
                lambda_upper = lyapunov_exponent + z_score * uncertainty
                
                # Lyapunov Time의 신뢰구간
                if lambda_lower > 0:
                    tau_upper = 1.0 / lambda_lower
                else:
                    tau_upper = np.inf
                    
                if lambda_upper > 0:
                    tau_lower = 1.0 / lambda_upper
                else:
                    tau_lower = 0.0
                
                result['confidence_interval'] = {
                    'lambda': (lambda_lower, lambda_upper),
                    'lyapunov_time': (tau_lower, tau_upper),
                    'confidence_level': 0.95
                }
                
            elif method == 'bootstrap':
                # Bootstrap 방법으로 신뢰구간 계산
                n_bootstrap = 1000
                bootstrap_times = []
                
                for _ in range(n_bootstrap):
                    # 정규분포를 가정한 부트스트랩 샘플
                    lambda_boot = np.random.normal(lyapunov_exponent, uncertainty)
                    if lambda_boot != 0:
                        tau_boot = 1.0 / abs(lambda_boot)
                        bootstrap_times.append(tau_boot)
                
                if bootstrap_times:
                    bootstrap_times = np.array(bootstrap_times)
                    # 95% 신뢰구간 (2.5%와 97.5% 백분위수)
                    tau_lower = np.percentile(bootstrap_times, 2.5)
                    tau_upper = np.percentile(bootstrap_times, 97.5)
                    
                    result['confidence_interval'] = {
                        'lyapunov_time': (tau_lower, tau_upper),
                        'bootstrap_samples': n_bootstrap,
                        'confidence_level': 0.95
                    }
        
        return result
    
    def analyze_prediction_horizon(self, trajectory, dt, methods=None, max_lag=None):
        """
        여러 방법을 사용하여 예측 가능 시간 분석
        
        Args:
            trajectory (np.ndarray): 시계열 데이터
            dt (float): 시간 간격
            methods (list): 사용할 Lyapunov 지수 계산 방법들
            max_lag (int): 최대 지연 시간
            
        Returns:
            dict: 각 방법별 예측 가능 시간 분석 결과
        """
        if methods is None:
            methods = ['linear_fit', 'wolf', 'rosenstein', 'kantz']
        
        analysis_results = {}
        
        for method in methods:
            try:
                # Lyapunov 지수 계산
                lyap_exp, _, uncertainty = self.calculate_lyapunov_exponent(
                    trajectory, dt, method=method, max_lag=max_lag
                )
                
                # Lyapunov Time 계산 (신뢰구간 포함)
                lyap_time_result = self.calculate_lyapunov_time(
                    lyap_exp, method='confidence_interval', uncertainty=uncertainty
                )
                
                analysis_results[method] = {
                    'lyapunov_exponent': lyap_exp,
                    'lyapunov_time': lyap_time_result['lyapunov_time'],
                    'prediction_horizon': lyap_time_result['prediction_horizon'],
                    'stability': lyap_time_result['stability'],
                    'uncertainty': uncertainty,
                    'confidence_interval': lyap_time_result.get('confidence_interval')
                }
                
            except Exception as e:
                analysis_results[method] = {
                    'error': str(e),
                    'lyapunov_exponent': None,
                    'lyapunov_time': None,
                    'prediction_horizon': None,
                    'stability': None
                }
        
        return analysis_results
    
    def calculate_autocorrelation(self, data, max_lag=None, normalize=True, method='statsmodels'):
        """
        자기상관 함수 계산
        
        Args:
            data (np.ndarray): 시계열 데이터 [time_steps, variables] 또는 [time_steps]
            max_lag (int): 최대 지연 시간 (None인 경우 전체 길이의 1/2)
            normalize (bool): 정규화 여부 (statsmodels에서는 항상 정규화됨)
            method (str): 계산 방법 ('statsmodels', 'scipy', 'manual')
            
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
            if method == 'statsmodels':
                try:
                    # statsmodels 사용 (권장)
                    from statsmodels.tsa.stattools import acf
                    autocorr = acf(data[:, i], nlags=max_lag, fft=True)
                except ImportError:
                    print("Warning: statsmodels not available, falling back to scipy method")
                    method = 'scipy'
            
            if method == 'scipy':
                # scipy.signal.correlate 사용 (기존 방법)
                autocorr = correlate(data[:, i], data[:, i], mode='full')
                autocorr = autocorr[n_steps-1:n_steps-1+max_lag+1]
                
                if normalize:
                    autocorr = autocorr / autocorr[0]
                    
            elif method == 'manual':
                # 수동 계산 (교육적 목적)
                autocorr = self._manual_autocorrelation(data[:, i], max_lag)
                
            all_autocorr.append(autocorr)
        
        # 평균 자기상관 함수
        mean_autocorr = np.mean(all_autocorr, axis=0)
        lags = np.arange(len(mean_autocorr))
        
        return lags, mean_autocorr
    
    def _manual_autocorrelation(self, data, max_lag):
        """
        수동으로 autocorrelation 계산 (교육적 목적)
        
        Args:
            data (np.ndarray): 1차원 시계열 데이터
            max_lag (int): 최대 지연 시간
            
        Returns:
            np.ndarray: autocorrelation 값들
        """
        n_steps = len(data)
        mean_val = np.mean(data)
        var_val = np.var(data)
        
        autocorr = np.zeros(max_lag + 1)
        
        for lag in range(max_lag + 1):
            if lag == 0:
                # lag=0: 항상 1.0
                autocorr[lag] = 1.0
            else:
                # lag>0: 시계열 상관계수 계산
                numerator = 0.0
                for t in range(n_steps - lag):
                    numerator += (data[t] - mean_val) * (data[t + lag] - mean_val)
                
                autocorr[lag] = numerator / ((n_steps - lag) * var_val)
        
        return autocorr
    
    def calculate_partial_autocorrelation(self, data, max_lag=None, method='statsmodels'):
        """
        부분 자기상관 함수(Partial Autocorrelation Function) 계산
        
        Args:
            data (np.ndarray): 시계열 데이터 [time_steps, variables] 또는 [time_steps]
            max_lag (int): 최대 지연 시간 (None인 경우 전체 길이의 1/2)
            method (str): 계산 방법 ('statsmodels', 'manual')
            
        Returns:
            tuple: (lags, pacf)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        n_steps, n_vars = data.shape
        
        if max_lag is None:
            max_lag = n_steps // 2
            
        max_lag = min(max_lag, n_steps - 1)
        
        # 각 변수별로 부분 자기상관 함수 계산
        all_pacf = []
        
        for i in range(n_vars):
            if method == 'statsmodels':
                try:
                    from statsmodels.tsa.stattools import pacf
                    pacf_values = pacf(data[:, i], nlags=max_lag, method='ols')
                    all_pacf.append(pacf_values)
                except ImportError:
                    print("Warning: statsmodels not available for PACF calculation")
                    return None, None
            else:
                # 수동 계산 (복잡하므로 statsmodels 권장)
                print("Manual PACF calculation not implemented. Using statsmodels.")
                try:
                    from statsmodels.tsa.stattools import pacf
                    pacf_values = pacf(data[:, i], nlags=max_lag, method='ols')
                    all_pacf.append(pacf_values)
                except ImportError:
                    return None, None
        
        # 평균 부분 자기상관 함수
        mean_pacf = np.mean(all_pacf, axis=0)
        lags = np.arange(len(mean_pacf))
        
        return lags, mean_pacf
    
    def calculate_autocorrelation_confidence_intervals(self, data, max_lag=None, confidence_level=0.95):
        """
        Autocorrelation의 신뢰구간 계산
        
        Args:
            data (np.ndarray): 시계열 데이터
            max_lag (int): 최대 지연 시간
            confidence_level (float): 신뢰수준 (기본값: 0.95)
            
        Returns:
            dict: 신뢰구간 정보
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        n_steps, n_vars = data.shape
        
        if max_lag is None:
            max_lag = n_steps // 2
            
        max_lag = min(max_lag, n_steps - 1)
        
        # Bartlett's formula를 사용한 신뢰구간 계산
        # 95% 신뢰구간: ±1.96/√N
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 99% 신뢰수준
        
        confidence_bound = z_score / np.sqrt(n_steps)
        
        # 각 변수별로 autocorrelation 계산
        lags, autocorr = self.calculate_autocorrelation(data, max_lag, method='statsmodels')
        
        if lags is None:
            return None
            
        confidence_intervals = {
            'lags': lags,
            'autocorr': autocorr,
            'upper_bound': confidence_bound,
            'lower_bound': -confidence_bound,
            'confidence_level': confidence_level,
            'sample_size': n_steps
        }
        
        return confidence_intervals
    
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
