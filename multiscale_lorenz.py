""" Lorenz-96 model
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18.
https://www.ecmwf.int/en/elibrary/10829-predictability-problem-partly-solved
"""

import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm
from statsmodels.tsa.stattools import acf

@njit
def L96_2t_xdot_ydot(X, Y, F, h, b, c):
    """
    Calculate the time rate of change for the X and Y variables for the Lorenz '96, two time-scale
    model, equations 2 and 3:
        d/dt X[k] =     -X[k-1] ( X[k-2] - X[k+1] )   - X[k] + F - h.c/b sum_j Y[j,k]
        d/dt Y[j] = -b c Y[j+1] ( Y[j+2] - Y[j-1] ) - c Y[j]     + h.c/b X[k]

    Args:
        X : Values of X variables at the current time step
        Y : Values of Y variables at the current time step
        F : Forcing term
        h : coupling coefficient
        b : ratio of amplitudes
        c : time-scale ratio
    Returns:
        dXdt, dYdt, C : Arrays of X and Y time tendencies, and the coupling term -hc/b*sum(Y,j)
    """

    JK, K = len(Y), len(X)
    J = JK // K
    assert JK == J * K, "X and Y have incompatible shapes"
    Xdot = np.zeros(K)
    hcb = (h * c) / b

    Ysummed = Y.reshape((K, J)).sum(axis=-1)

    Xdot = np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2)) - X + F - hcb * Ysummed
    Ydot = (
        -c * b * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1))
        - c * Y
        + hcb * np.repeat(X, J)
    )

    return Xdot, Ydot, -hcb * Ysummed

# Time-stepping methods ##########################################################################################
def RK4(
    X: np.ndarray, 
    Y: np.ndarray, 
    dt: float, 
    F: float, 
    h: float, 
    b: float, 
    c: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """단일 RK4 스텝을 수행합니다.

    Args:
        X: X 변수의 현재 상태
        Y: Y 변수의 현재 상태
        dt: 시간 간격
        F: 외력 항
        h: 결합 계수
        b: 진폭 비율
        c: 시간 스케일 비율

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 새로운 X 상태, Y 상태, 그리고 결합항
    """
    k1x, k1y, XYtend = L96_2t_xdot_ydot(X, Y, F, h, b, c)
    k2x, k2y, _ = L96_2t_xdot_ydot(
        X + 0.5 * dt * k1x, Y + 0.5 * dt * k1y, F, h, b, c
    )
    k3x, k3y, _ = L96_2t_xdot_ydot(
        X + 0.5 * dt * k2x, Y + 0.5 * dt * k2y, F, h, b, c
    )
    k4x, k4y, _ = L96_2t_xdot_ydot(
        X + dt * k3x, Y + dt * k3y, F, h, b, c
    )
    
    X_new = X + (dt / 6.0) * ((k1x + k4x) + 2.0 * (k2x + k3x))
    Y_new = Y + (dt / 6.0) * ((k1y + k4y) + 2.0 * (k2y + k3y))
    
    return X_new, Y_new, XYtend


# Model integrators #############################################################################################
def integrate_L96_2t_with_coupling(X0, Y0, si, nt, F, h, b, c, t0=0, dt=0.001):
    """
    Integrates forward-in-time the two time-scale Lorenz 1996 model, using the RK4 integration method.
    Returns the full history with nt+1 values starting with initial conditions, X[:,0]=X0 and Y[:,0]=Y0,
    and ending with the final state, X[:,nt+1] and Y[:,nt+1] at time t0+nt*si.

    Note the model is intergrated

    Args:
        X0 : Values of X variables at the current time
        Y0 : Values of Y variables at the current time
        si : Sampling time interval
        nt : Number of sample segments (results in nt+1 samples incl. initial state)
        F  : Forcing term
        h  : coupling coefficient
        b  : ratio of amplitudes
        c  : time-scale ratio
        t0 : Initial time (defaults to 0)
        dt : The actual time step. If dt<si, then si is used. Otherwise si/dt must be a whole number. Default 0.001.

    Returns:
        X[:,:], Y[:,:], time[:], hcbY[:,:] : the full history X[n,k] and Y[n,k] at times t[n], and coupling term

    Example usage:
        X,Y,t,_ = integrate_L96_2t_with_coupling(5+5*np.random.rand(8), np.random.rand(8*4), 0.01, 500, 18, 1, 10, 10)
        plt.plot( t, X);
    """

    time, xhist, yhist, xytend_hist = (
        t0 + np.zeros((nt + 1)),
        np.zeros((nt + 1, len(X0))),
        np.zeros((nt + 1, len(Y0))),
        np.zeros((nt + 1, len(X0))),
    )
    X, Y = X0.copy(), Y0.copy()
    xhist[0, :] = X
    yhist[0, :] = Y
    xytend_hist[0, :] = 0
    if si < dt:
        dt, ns = si, 1
    else:
        ns = int(si / dt + 0.5)
        assert (
            abs(ns * dt - si) < 1e-14
        ), "si is not an integer multiple of dt: si=%f dt=%f ns=%i" % (si, dt, ns)

    for n in range(nt):
        for s in range(ns):
            X, Y, XYtend = RK4(X, Y, dt, F, h, b, c)

        xhist[n + 1], yhist[n + 1], time[n + 1], xytend_hist[n + 1] = (
            X,
            Y,
            t0 + si * (n + 1),
            XYtend,
        )
    return xhist, yhist, time, xytend_hist


# Adaptive Time-Stepping Model integrators #############################################################################################
def calculate_normalized_error(
    X_full: np.ndarray, 
    Y_full: np.ndarray,
    X_half: np.ndarray, 
    Y_half: np.ndarray
) -> float:
    """정규화된 오차를 계산합니다.

    Args:
        X_full: 전체 스텝으로 계산된 X
        Y_full: 전체 스텝으로 계산된 Y
        X_half: 절반 스텝으로 계산된 X
        Y_half: 절반 스텝으로 계산된 Y

    Returns:
        float: 정규화된 최대 상대 오차
    """
    error_x = np.linalg.norm(X_full - X_half) / (np.linalg.norm(X_full) + 1e-15)
    error_y = np.linalg.norm(Y_full - Y_half) / (np.linalg.norm(Y_full) + 1e-15)
    return max(error_x, error_y)


def adjust_timestep(
    dt: float, 
    error: float, 
    tol: float, 
    dt_max: float = 0.001, 
    dt_min: float = 1e-6,
    safety_factor: float = 0.9
) -> float:
    """오차에 기반하여 시간 간격을 조절합니다.

    Args:
        dt: 현재 시간 간격
        error: 계산된 오차
        tol: 허용 오차
        dt_max: 최대 허용 시간 간격 (기본값: 0.001)
        dt_min: 최소 허용 시간 간격 (기본값: 1e-6)
        safety_factor: 안전 계수 (기본값: 0.9)

    Returns:
        float: 조절된 새로운 시간 간격
    """
    if error < tol:
        # PI 제어기 기반 시간 간격 증가
        dt_new = dt * min(2.0, safety_factor * (tol/error)**0.2)
        return min(dt_max, dt_new)
    else:
        # PI 제어기 기반 시간 간격 감소
        dt_new = dt * max(0.1, safety_factor * (tol/error)**0.25)
        return max(dt_min, dt_new)


def adaptive_integrate_L96_2t_with_coupling(
    X0: np.ndarray,
    Y0: np.ndarray,
    si: float,
    nt: int,
    F: float,
    h: float,
    b: float,
    c: float,
    t0: float = 0,
    dt: float = 0.001,
    tol: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """적응형 RK4 방법을 사용하여 Lorenz 96 two-timescale 모델을 적분합니다.

    Args:
        X0: X 변수의 초기 조건
        Y0: Y 변수의 초기 조건
        si: 샘플링 간격
        nt: 샘플링 횟수
        F: 외력 항
        h: 결합 계수
        b: 진폭 비율
        c: 시간 스케일 비율
        t0: 초기 시간 (기본값: 0)
        dt: 초기 시간 간격 (기본값: 0.001)
        tol: 허용 오차 (기본값: 1e-6)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            - X 변수의 시간 이력
            - Y 변수의 시간 이력
            - 시간 배열
            - 결합항의 시간 이력
    """
    # 결과 저장을 위한 배열 초기화
    time = t0 + np.zeros(nt + 1)
    xhist = np.zeros((nt + 1, len(X0)))
    yhist = np.zeros((nt + 1, len(Y0)))
    xytend_hist = np.zeros((nt + 1, len(X0)))
    
    # 초기 조건 저장
    X, Y = X0.copy(), Y0.copy()
    xhist[0], yhist[0] = X, Y
    time[0] = t0
    xytend_hist[0] = 0
    current_time = t0
    
    for n in range(nt):
        target_time = t0 + (n + 1) * si
        
        while current_time < target_time:
            # 남은 시간이 dt보다 작으면 조정
            dt = min(dt, target_time - current_time)
            
            # 전체 스텝 계산
            X_full, Y_full, XYtend = RK4(X, Y, dt, F, h, b, c)
            
            # 절반 스텝 계산
            dt_half = dt / 2
            X_mid, Y_mid, _ = RK4(X, Y, dt_half, F, h, b, c)
            X_half, Y_half, _ = RK4(X_mid, Y_mid, dt_half, F, h, b, c)
            
            # 오차 계산
            error = calculate_normalized_error(X_full, Y_full, X_half, Y_half)
            
            # 수치적 불안정성 검사
            if np.any(np.isnan(X_full)) or np.any(np.isnan(Y_full)):
                raise RuntimeError("수치적 불안정성 발생: NaN 값 감지")
            
            # 시간 간격 조절 및 결과 업데이트
            new_dt = adjust_timestep(dt, error, tol)
            
            if error < tol:
                X, Y = X_full, Y_full
                current_time += dt
                dt = new_dt
            else:
                dt = new_dt
                continue
        
        # 샘플링 시점의 결과 저장
        xhist[n + 1] = X
        yhist[n + 1] = Y
        time[n + 1] = current_time
        xytend_hist[n + 1] = XYtend
        
    return xhist, yhist, time, xytend_hist


def s(k, K):
    """A non-dimension coordinate from -1..+1 corresponding to k=0..K"""
    return 2 * (0.5 + k) / K - 1


# Class for convenience
class L96:
    """Two time-scale Lorenz 1996 model with adaptive time-stepping capability.
    
    Attributes:
        X (np.ndarray): Current X state or initial conditions
        Y (np.ndarray): Current Y state or initial conditions
        F (float): Forcing term
        h (float): Coupling coefficient
        b (float): Ratio of amplitudes
        c (float): Time-scale ratio
        dt (float): Time step
        K (int): Number of X values
        J (int): Number of Y values per X value
        JK (int): Total number of Y values (J * K)
    """

    def __init__(self, K: int, J: int, F: float = 18, h: float = 1, 
                 b: float = 10, c: float = 10, t: float = 0, dt: float = 0.001):
        """Construct a two time-scale model with parameters.

        Args:
            K: Number of X values
            J: Number of Y values per X value
            F: Forcing term (default 18.)
            h: Coupling coefficient (default 1.)
            b: Ratio of amplitudes (default 10.)
            c: Time-scale ratio (default 10.)
            t: Initial time (default 0.)
            dt: Time step (default 0.001)
        """
        self.F, self.h, self.b, self.c, self.dt = F, h, b, c, dt
        self.t = t
        self.K, self.J = K, J
        self.JK = J * K
        
        # Initialize states with random values
        self.X = b * np.random.randn(K)
        self.Y = np.random.randn(self.JK)
        
        # For plotting convenience
        self.k = np.arange(self.K)
        self.j = np.arange(self.JK)

    def __repr__(self) -> str:
        return (f"L96: K={self.K} J={self.J} F={self.F} h={self.h} "
                f"b={self.b} c={self.c} dt={self.dt}")

    def __str__(self) -> str:
        return f"{self.__repr__()}\nX={self.X}\nY={self.Y}\nt={self.t}"

    def copy(self):
        """Create a deep copy of the current model instance."""
        copy = L96(self.K, self.J, F=self.F, h=self.h, b=self.b, c=self.c, dt=self.dt)
        copy.set_state(self.X.copy(), self.Y.copy(), t=self.t)
        return copy

    def set_param(self, **kwargs):
        """Set model parameters.
        
        Args:
            **kwargs: Parameters to update (dt, F, h, b, c, or t)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def set_state(self, X: np.ndarray, Y: np.ndarray, t: float = None):
        """Set model state.
        
        Args:
            X: New X state
            Y: New Y state
            t: New time value (optional)
        """
        self.X, self.Y = X, Y
        if t is not None:
            self.t = t
        return self

    def randomize_IC(self):
        """Randomize the initial conditions."""
        self.X = self.b * np.random.rand(self.K)
        self.Y = np.random.rand(self.JK)
        return self

    def run(self, si: float, T: float, store: bool = False, 
            return_coupling: bool = False, adaptive: bool = True, 
            tol: float = 1e-6) -> tuple:
        """Run model integration.
        
        Args:
            si: Sampling interval
            T: Total integration time
            store: Whether to store final state
            return_coupling: Whether to return coupling term
            adaptive: Whether to use adaptive time-stepping
            tol: Error tolerance for adaptive stepping
            
        Returns:
            tuple: (X history, Y history, time points, [coupling term])
        """
        nt = int(T / si)
        
        if adaptive:
            X, Y, t, C = adaptive_integrate_L96_2t_with_coupling(
                self.X, self.Y, si, nt, self.F, self.h, self.b, self.c,
                t0=self.t, dt=self.dt, tol=tol
            )
        else:
            X, Y, t, C = integrate_L96_2t_with_coupling(
                self.X, self.Y, si, nt, self.F, self.h, self.b, self.c,
                t0=self.t, dt=self.dt
            )
        
        if store:
            self.X, self.Y, self.t = X[-1], Y[-1], t[-1]
            
        return (X, Y, t, C) if return_coupling else (X, Y, t)


    
if __name__ == "__main__":
    # Arnold's experimental setup
    K = 8       # Number of X variables (low-frequency, large-amplitude)
    J = 32      # Number of Y variables per X (high-frequency, small-amplitude)
    F = 20.0    # Forcing
    h = 1.0     # Coupling coefficient
    b = 10.0    # Ratio of amplitudes
    
    # 두 가지 time-scale ratio 설정
    c_values = [4.0, 10.0]
    
    # 적분 파라미터
    dt = 0.001  # Maximum time step (MTU)
    si = 0.001  # Sampling interval
    spinup_time = 100.0  # Spin-up time to remove transients
    
    # 자기상관 분석을 위한 긴 시간 적분
    analysis_time = 50.0  # 분석을 위한 긴 시간 적분 (MTU)
    
    print("\n=== Temporal Autocorrelation Analysis ===")
    
    plt.figure(figsize=(15, 10))
    for i, c in enumerate(c_values):
        print(f"\n분석 중: time-scale ratio c = {c}")
        
        # 모델 초기화 및 스핀업
        model = L96(K, J, F=F, h=h, b=b, c=c, dt=dt)
        print("스핀업 수행 중...")
        X_spinup, Y_spinup, t_spinup = model.run(si, spinup_time, store=True, adaptive=True, tol=1e-6)
        
        # 분석을 위한 긴 시간 적분
        print("긴 시간 적분 수행 중...")
        X_analysis, Y_analysis, t_analysis = model.run(si, analysis_time, store=True, adaptive=True, tol=1e-6)
        
        # 시계열 자기상관 계산
        max_lag = int(30.0 / si)  # 30 MTU까지의 lag 계산
        autocorr_all = np.zeros((K, max_lag+1))
        
        # 개별 변수별 자기상관 계산
        plt.subplot(2, 2, i+1)
        for k in tqdm(range(K)):
            # statsmodels의 acf 함수 사용
            autocorr_all[k] = acf(X_analysis[:, k], nlags=max_lag, fft=True)
            # 개별 변수의 자기상관 플롯 (연한 색상)
            plt.plot(np.arange(max_lag+1) * si, autocorr_all[k], alpha=0.3, 
                    label=f'X_{k+1}')
            
        # K개 변수의 평균 자기상관
        autocorr = np.mean(autocorr_all, axis=0)
        
        # 평균 자기상관 플롯 (굵은 검은색 선)
        plt.plot(np.arange(max_lag+1) * si, autocorr, 'k-', linewidth=2, label='Mean')
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Lag (MTU)')
        plt.ylabel('Autocorrelation')
        plt.title(f'Individual Autocorrelations (c = {c})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 평균 자기상관만 따로 표시
        plt.subplot(2, 2, i+3)
        plt.plot(np.arange(max_lag+1) * si, autocorr, 'b-', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('Lag (MTU)')
        plt.ylabel('Mean Autocorrelation')
        plt.title(f'Mean Autocorrelation (c = {c})')
        
        # 주요 시점들의 자기상관 값 표시
        important_lags = [5, 10, 15, 20]
        for lag in important_lags:
            lag_idx = int(lag / si)
            plt.plot(lag, autocorr[lag_idx], 'ro')
            # 5, 15 MTU는 위쪽에, 10, 20 MTU는 아래쪽에 표시
            y_offset = 0.3 if lag in [5, 15] else -0.3
            plt.annotate(f'r({lag} MTU) = {autocorr[lag_idx]:.3f}', 
                        xy=(lag, autocorr[lag_idx]), 
                        xytext=(lag + 2, autocorr[lag_idx] + y_offset),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5),
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 자기상관 분석 결과를 바탕으로 실제 실험 수행
    interval = 10.0  # 자기상관 분석 결과를 바탕으로 선택된 간격 (MTU)
    num_ic = 300    # Number of initial conditions
    forecast_time = 0.2  # Forecast time for each run (MTU)
    
    # 결과 저장을 위한 딕셔너리
    results = {}
    print("\n=== Running Main Experiment ===")
    for c in c_values:
        print(f"\n실험 수행 중: time-scale ratio c = {c}")
        
        # 모델 초기화 및 스핀업
        model = L96(K, J, F=F, h=h, b=b, c=c, dt=dt)
        print("스핀업 수행 중...")
        X_spinup, Y_spinup, t_spinup = model.run(si, spinup_time, store=True, adaptive=True, tol=1e-6)
        
        # 300개의 초기 조건 생성 및 적분
        X_all = []
        t_all = []
        
        print(f"{num_ic}개의 예측 수행 중...")
        for i in tqdm(range(num_ic)):
            # 각 초기 조건에서 예측 수행
            X, Y, t = model.run(si, interval, store=True, adaptive=True, tol=1e-6)
            X_forecast, Y_forecast, t_forecast = model.run(si, forecast_time, store=True, adaptive=True, tol=1e-6)
            
            X_all.append(X_forecast)
            t_all.append(t_forecast)
        
        results[c] = {
            'X': np.array(X_all),
            't': np.array(t_all)
        }
    
    # 상태 공간 플롯 (X1 vs X2)
    plt.figure(figsize=(12, 5))
    for i, c in enumerate(c_values):
        plt.subplot(1, 2, i+1)
        plt.plot(results[c]['X'][:, :, 0].flatten(), 
                results[c]['X'][:, :, 1].flatten(), 
                '.', alpha=0.1, markersize=1)
        plt.xlabel('X₁')
        plt.ylabel('X₂')
        plt.title(f'State Space Plot (c = {c})')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()