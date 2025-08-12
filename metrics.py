import numpy as np
import matplotlib.pyplot as plt

class ClimateMetrics:
    """
    기후 예측 모델의 성능을 평가하는 다양한 지표들을 계산하는 클래스
    
    포함된 지표들:
    1. Mean state error (평균 상태 오차)
    2. Variance ratio (분산 비율)
    3. Kullback-Leibler divergence (쿨백-라이블러 발산)
    4. Extreme event frequency (극단 이벤트 빈도)
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
    
    def calculate_all_metrics(self, u_pred, u_true, threshold_std=2.0):
        """
        모든 평가 지표를 한 번에 계산
        
        Args:
            u_pred (np.ndarray): 예측값 [time_steps, variables] 또는 [time_steps]
            u_true (np.ndarray): 실제값 [time_steps, variables] 또는 [time_steps]
            threshold_std (float): 극단 이벤트 임계값
            
        Returns:
            dict: 모든 평가 지표를 포함한 딕셔너리
        """
        metrics = {}
        
        # Mean state error
        metrics['mean_state_error'] = self.mean_state_error(u_pred, u_true)
        metrics['mean_state_error_mean'] = np.mean(metrics['mean_state_error'])
        
        # Variance ratio
        metrics['variance_ratio'] = self.variance_ratio(u_pred, u_true)
        
        # KL divergence
        metrics['kl_divergence'] = self.kullback_leibler_divergence(u_pred, u_true)
        
        # Extreme event frequency (예측값과 실제값 각각)
        metrics['extreme_event_freq_pred'] = self.extreme_event_frequency(u_pred, threshold_std)
        metrics['extreme_event_freq_true'] = self.extreme_event_frequency(u_true, threshold_std)
        
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
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Climate Metrics Over Time', fontsize=16)
        
        # Mean State Error
        axes[0, 0].plot(time_axis, metrics['mean_state_error'])
        axes[0, 0].set_title('Mean State Error')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Error')
        axes[0, 0].grid(True)
        
        # Variance Ratio (상수이므로 수평선)
        axes[0, 1].axhline(y=metrics['variance_ratio'], color='r', linestyle='-', label=f'Variance Ratio: {metrics["variance_ratio"]:.4f}')
        axes[0, 1].set_title('Variance Ratio')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # KL Divergence (상수이므로 수평선)
        axes[1, 0].axhline(y=metrics['kl_divergence'], color='g', linestyle='-', label=f'KL Divergence: {metrics["kl_divergence"]:.4f}')
        axes[1, 0].set_title('KL Divergence')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Divergence')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Extreme Event Frequencies
        axes[1, 1].bar(['Prediction', 'True'], 
                       [metrics['extreme_event_freq_pred'], metrics['extreme_event_freq_true']],
                       color=['blue', 'orange'])
        axes[1, 1].set_title('Extreme Event Frequency')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"지표 그래프가 {save_path}에 저장되었습니다.")
        
        plt.show()
