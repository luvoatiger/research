#!/usr/bin/env python3
"""
Lyapunov 지수, 자기상관 함수, PDF 계산 기능을 테스트하는 스크립트
"""

import numpy as np
import matplotlib.pyplot as plt
from metrics import ClimateMetrics

def generate_lorenz96_data(n_steps=10000, dt=0.005, F=8.0):
    """
    Lorenz 96 시스템의 시뮬레이션 데이터 생성 (간단한 근사)
    
    Args:
        n_steps (int): 시간 스텝 수
        dt (float): 시간 간격
        F (float): 강제력 파라미터
        
    Returns:
        tuple: (time, X_data)
    """
    print(f"Lorenz 96 데이터 생성 중... (n_steps={n_steps}, dt={dt})")
    
    # 초기 조건
    K = 8
    X = np.zeros((n_steps, K))
    X[0] = np.random.uniform(-2, 2, K)
    
    # 간단한 Euler 적분으로 Lorenz 96 시스템 시뮬레이션
    for i in range(1, n_steps):
        for k in range(K):
            # 순환 경계 조건
            k_plus_1 = (k + 1) % K
            k_minus_1 = (k - 1) % K
            k_minus_2 = (k - 2) % K
            
            # Lorenz 96 방정식
            dX_dt = (X[i-1, k_plus_1] - X[i-1, k_minus_2]) * X[i-1, k_minus_1] - X[i-1, k] + F
            X[i, k] = X[i-1, k] + dt * dX_dt
    
    time = np.arange(n_steps) * dt
    return time, X

def test_lyapunov_exponent():
    """Lyapunov 지수 계산 테스트"""
    print("\n" + "="*60)
    print("           LYAPUNOV EXPONENT TEST")
    print("="*60)
    
    # 데이터 생성
    time, X_data = generate_lorenz96_data(n_steps=5000, dt=0.005)
    
    # ClimateMetrics 인스턴스 생성
    climate_metrics = ClimateMetrics()
    
    # Lyapunov 지수 계산 (두 가지 방법)
    print("\n1. Linear Fit 방법으로 Lyapunov 지수 계산:")
    lyap_exp1, lyap_time1, lyap_unc1 = climate_metrics.calculate_lyapunov_exponent(
        X_data, dt=0.005, method='linear_fit'
    )
    print(f"   Lyapunov 지수: {lyap_exp1:.6f} ± {lyap_unc1:.6f}")
    print(f"   Lyapunov 시간: {lyap_time1:.6f}")
    
    print("\n2. Wolf 방법으로 Lyapunov 지수 계산:")
    lyap_exp2, lyap_time2, lyap_unc2 = climate_metrics.calculate_lyapunov_exponent(
        X_data, dt=0.005, method='wolf'
    )
    print(f"   Lyapunov 지수: {lyap_exp2:.6f} ± {lyap_unc2:.6f}")
    print(f"   Lyapunov 시간: {lyap_time2:.6f}")
    
    return X_data

def test_autocorrelation():
    """자기상관 함수 계산 테스트"""
    print("\n" + "="*60)
    print("           AUTOCORRELATION FUNCTION TEST")
    print("="*60)
    
    # 데이터 생성
    time, X_data = generate_lorenz96_data(n_steps=3000, dt=0.005)
    
    # ClimateMetrics 인스턴스 생성
    climate_metrics = ClimateMetrics()
    
    # 자기상관 함수 계산
    print("\n자기상관 함수 계산 중...")
    lags, autocorr = climate_metrics.calculate_autocorrelation(X_data, max_lag=100)
    
    print(f"   지연 시간 범위: 0 ~ {lags[-1] * 0.005:.3f} time units")
    print(f"   자기상관 함수 길이: {len(autocorr)}")
    
    # 자기상관 함수 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(lags * 0.005, autocorr, 'b-', linewidth=2)
    plt.title('Autocorrelation Function of Lorenz 96 System')
    plt.xlabel('Lag (time units)')
    plt.ylabel('Autocorrelation')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('autocorrelation_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_data

def test_pdf():
    """확률 밀도 함수 계산 테스트"""
    print("\n" + "="*60)
    print("           PROBABILITY DENSITY FUNCTION TEST")
    print("="*60)
    
    # 데이터 생성
    time, X_data = generate_lorenz96_data(n_steps=5000, dt=0.005)
    
    # ClimateMetrics 인스턴스 생성
    climate_metrics = ClimateMetrics()
    
    # PDF 계산
    print("\nPDF 계산 중...")
    bin_centers, pdf_values = climate_metrics.calculate_pdf(X_data, bins=100)
    
    print(f"   빈 수: {len(bin_centers)}")
    print(f"   데이터 범위: [{bin_centers.min():.3f}, {bin_centers.max():.3f}]")
    
    # PDF 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, pdf_values, 'r-', linewidth=2)
    plt.title('Probability Density Function of Lorenz 96 System (X1)')
    plt.xlabel('X1')
    plt.ylabel('Probability density')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pdf_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_data

def test_all_metrics():
    """모든 지표를 한 번에 계산하는 테스트"""
    print("\n" + "="*60)
    print("           ALL METRICS INTEGRATION TEST")
    print("="*60)
    
    # 데이터 생성
    time, X_data = generate_lorenz96_data(n_steps=4000, dt=0.005)
    
    # 예측값과 실제값으로 분할 (간단한 시뮬레이션)
    split_point = len(X_data) // 2
    X_true = X_data[:split_point]
    X_pred = X_data[split_point:] + 0.1 * np.random.randn(*X_data[split_point:].shape)  # 노이즈 추가
    
    # ClimateMetrics 인스턴스 생성
    climate_metrics = ClimateMetrics()
    
    # 모든 지표 계산
    print("\n모든 지표 계산 중...")
    metrics = climate_metrics.calculate_all_metrics(X_pred, X_true, dt=0.005)
    
    # 결과 출력
    climate_metrics.print_metrics_summary(metrics)
    
    # 시각화
    save_path = "all_metrics_test.png"
    climate_metrics.plot_metrics_over_time(metrics, save_path=save_path)
    
    # 기후 통계 비교 그래프
    save_path_climate = "climate_statistics_test.png"
    climate_metrics.plot_climate_statistics(metrics, save_path=save_path_climate)
    
    return metrics

def main():
    """메인 테스트 함수"""
    print("Lyapunov 지수, 자기상관 함수, PDF 계산 기능 테스트")
    print("="*60)
    
    try:
        # 1. Lyapunov 지수 테스트
        X_data1 = test_lyapunov_exponent()
        
        # 2. 자기상관 함수 테스트
        X_data2 = test_autocorrelation()
        
        # 3. PDF 테스트
        X_data3 = test_pdf()
        
        # 4. 통합 테스트
        metrics = test_all_metrics()
        
        print("\n" + "="*60)
        print("           ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("생성된 파일들:")
        print("- autocorrelation_test.png")
        print("- pdf_test.png")
        print("- all_metrics_test.png")
        print("- climate_statistics_test.png")
        
    except Exception as e:
        print(f"\n테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 