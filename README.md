# Research Project: Lorenz 96 System with dAMZ Neural Network

## 프로젝트 개요
이 프로젝트는 Lorenz 96 시스템을 dAMZ (discrete Approximated Mori Zwanzig) 신경망으로 모델링하고, 기후 예측 모델의 성능을 다양한 지표로 평가하는 연구입니다.

## 주요 구성 요소

### 1. dAMZ (discrete Approximated Mori Zwanzig) 신경망
- **구조**: LSTM 기반 신경망으로 변경됨 (기존 MLP에서 업그레이드)
- **목적**: 복잡한 동적 시스템을 저차원으로 근사화
- **특징**: 
  - Markov term: Lorenz 96 시스템의 실제 동역학 방정식
  - Memory term: LSTM 신경망으로 학습
  - 시퀀스 데이터의 시간적 의존성 포착

### 2. ClimateMetrics 클래스
기후 예측 모델의 성능을 평가하는 포괄적인 지표들을 제공합니다.

#### 기본 지표
- **Mean State Error**: 예측값과 실제값 간의 평균 상태 오차
- **Variance Ratio**: 예측값과 실제값의 분산 비율
- **Kullback-Leibler Divergence**: 확률 분포 간의 차이
- **Extreme Event Frequency**: 극단 이벤트 발생 빈도

#### 고급 지표 (새로 추가됨)
- **Lyapunov Exponent**: 카오스 시스템의 민감도 측정
  - Linear Fit 방법
  - Wolf 방법
  - 불확실성 추정 포함
- **Lyapunov Time**: 예측 가능한 시간 스케일
- **Autocorrelation Function**: 시간적 상관관계 분석
- **Probability Density Function**: 확률 분포 비교

### 3. 시각화 기능
- **기본 지표 시각화**: 시간에 따른 지표 변화
- **기후 통계 비교**: Figure 4.3과 유사한 PDF 및 Autocorrelation 비교
- **Lyapunov 지수 분석**: 예측 모델과 실제 시스템의 카오스 특성 비교

## 파일 구조

```
research/
├── memory.py              # 메인 dAMZ 모델 및 시뮬레이션 함수
├── metrics.py             # ClimateMetrics 클래스 (모든 평가 지표)
├── test_lyapunov_metrics.py  # 새로운 기능 테스트 스크립트
├── baseline.py            # 베이스라인 모델
├── multiscale_lorenz.py  # Lorenz 96 시스템 구현
├── simulated_data/        # 시뮬레이션 데이터
└── README.md             # 이 파일
```

## 사용 방법

### 1. 기본 실행
```bash
python memory.py
```

### 2. 새로운 기능 테스트
```bash
python test_lyapunov_metrics.py
```

### 3. 개별 기능 사용 예시
```python
from metrics import ClimateMetrics

# ClimateMetrics 인스턴스 생성
climate_metrics = ClimateMetrics()

# Lyapunov 지수 계산
lyap_exp, lyap_time, lyap_unc = climate_metrics.calculate_lyapunov_exponent(
    trajectory_data, dt=0.005, method='linear_fit'
)

# 자기상관 함수 계산
lags, autocorr = climate_metrics.calculate_autocorrelation(data)

# PDF 계산
bin_centers, pdf_values = climate_metrics.calculate_pdf(data)

# 모든 지표 계산
metrics = climate_metrics.calculate_all_metrics(u_pred, u_true, dt=0.005)
```

## 주요 특징

### LSTM 기반 아키텍처
- **입력 처리**: 시퀀스 데이터를 [batch_size, n_M+1, d] 형태로 재구성
- **메모리 효과**: 장기 의존성 학습
- **가중치 초기화**: LSTM과 Linear 레이어에 최적화된 초기화

### 고급 분석 기능
- **카오스 분석**: Lyapunov 지수를 통한 시스템 안정성 평가
- **시간적 특성**: Autocorrelation을 통한 메모리 효과 분석
- **통계적 특성**: PDF를 통한 분포 특성 비교

### 통합 평가 시스템
- **일관된 인터페이스**: 모든 지표를 하나의 클래스에서 제공
- **자동 시각화**: 계산된 지표들의 자동 그래프 생성
- **불확실성 추정**: Monte Carlo 방법을 통한 예측 신뢰도 평가

## 의존성

- Python 3.7+
- NumPy
- Matplotlib
- PyTorch
- SciPy

## 참고 문헌

- Lorenz 96 시스템: 대기 과학에서 사용되는 카오스 시스템
- Mori-Zwanzig 분해: 동적 시스템의 마르코프 항과 메모리 항 분리
- Lyapunov 지수: 카오스 시스템의 민감도 측정 지표

## 라이센스

이 프로젝트는 연구 목적으로 개발되었습니다.