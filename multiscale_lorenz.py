""" Lorenz-96 model
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18.
https://www.ecmwf.int/en/elibrary/10829-predictability-problem-partly-solved
"""

import os
import numpy as np

from numba import jit, njit
from tqdm import tqdm


@njit
def L96_eq1_xdot(X, F, advect=True):
    """
    Calculate the time rate of change for the X variables for the Lorenz '96, equation 1:
        d/dt X[k] = -X[k-2] X[k-1] + X[k-1] X[k+1] - X[k] + F

    Args:
        X : Values of X variables at the current time step
        F : Forcing term
    Returns:
        dXdt : Array of X time tendencies
    """

    K = len(X)
    Xdot = np.zeros(K)

    if advect:
        Xdot = np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2)) - X + F
    else:
        Xdot = -X + F
    #     for k in range(K):
    #         Xdot[k] = ( X[(k+1)%K] - X[k-2] ) * X[k-1] - X[k] + F
    return Xdot


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
    hcb = (h * c) / J

    Ysummed = Y.reshape((K, J)).sum(axis=-1)

    Xdot = np.roll(X, 1) * (np.roll(X, -1) - np.roll(X, 2)) - X + F - h * Ysummed / J
    Ydot = (
        -c * J * np.roll(Y, -1) * (np.roll(Y, -2) - np.roll(Y, 1))
        - c * Y
        + hcb * np.repeat(X, J)
    )

    return Xdot, Ydot, - h * Ysummed / J


# Time-stepping methods ##########################################################################################


def EulerFwd(fn, dt, X, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,F) using the Euler forward method.

    Args:
        fn : The function returning the time rate of change of model variables X
        dt : The time step
        X  : Values of X variables at the current time, t
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)

    Returns:
        X at t+dt
    """

    return X + dt * fn(X, *params)


def RK2(fn, dt, X, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,F) using the second order Runge-Kutta method.

    Args:
        fn : The function returning the time rate of change of model variables X
        dt : The time step
        X  : Values of X variables at the current time, t
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)

    Returns:
        X at t+dt
    """

    X1 = X + 0.5 * dt * fn(X, *params)
    return X + dt * fn(X1, *params)


def RK4(fn, dt, X, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,...) using the fourth order Runge-Kutta method.

    Args:
        fn     : The function returning the time rate of change of model variables X
        dt     : The time step
        X      : Values of X variables at the current time, t
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)

    Returns:
        X at t+dt
    """

    Xdot1 = fn(X, *params)
    Xdot2 = fn(X + 0.5 * dt * Xdot1, *params)
    Xdot3 = fn(X + 0.5 * dt * Xdot2, *params)
    Xdot4 = fn(X + dt * Xdot3, *params)
    return X + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))


# Model integrators #############################################################################################


# @jit(forceobj=True)
def integrate_L96_1t(X0, F, dt, nt, method=RK4, t0=0):
    """
    Integrates forward-in-time the single time-scale Lorenz 1996 model, using the integration "method".
    Returns the full history with nt+1 values starting with initial conditions, X[:,0]=X0, and ending
    with the final state, X[:,nt+1] at time t0+nt*dt.

    Args:
        X0     : Values of X variables at the current time
        F      : Forcing term
        dt     : The time step
        nt     : Number of forwards steps
        method : The time-stepping method that returns X(n+1) given X(n)
        t0     : Initial time (defaults to 0)

    Returns:
        X[:,:], time[:] : the full history X[n,k] at times t[n]

    Example usage:
        X,t = integrate_L96_1t(5+5*np.random.rand(8), 18, 0.01, 500)
        plt.plot(t, X);
    """

    time, hist = t0 + np.zeros((nt + 1)), np.zeros((nt + 1, len(X0)))
    X = X0.copy()
    hist[0, :] = X
    for n in range(nt):
        X = method(L96_eq1_xdot, dt, X, F)
        hist[n + 1], time[n + 1] = X, t0 + dt * (n + 1)
    return hist, time


# @jit(forceobj=True)
def integrate_L96_2t(X0, Y0, si, nt, F, h, b, c, t0=0, dt=0.001):
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
        X[:,:], Y[:,:], time[:] : the full history X[n,k] and Y[n,k] at times t[n]

    Example usage:
        X,Y,t = integrate_L96_2t(5+5*np.random.rand(8), np.random.rand(8*4), 0.01, 500, 18, 1, 10, 10)
        plt.plot( t, X);
    """

    xhist, yhist, time, _ = integrate_L96_2t_with_coupling(
        X0, Y0, si, nt, F, h, b, c, t0=t0, dt=dt
    )

    return xhist, yhist, time


# @jit(forceobj=True)
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
            # RK4 update of X,Y
            Xdot1, Ydot1, XYtend = L96_2t_xdot_ydot(X, Y, F, h, b, c)
            Xdot2, Ydot2, _ = L96_2t_xdot_ydot(
                X + 0.5 * dt * Xdot1, Y + 0.5 * dt * Ydot1, F, h, b, c
            )
            Xdot3, Ydot3, _ = L96_2t_xdot_ydot(
                X + 0.5 * dt * Xdot2, Y + 0.5 * dt * Ydot2, F, h, b, c
            )
            Xdot4, Ydot4, _ = L96_2t_xdot_ydot(
                X + dt * Xdot3, Y + dt * Ydot3, F, h, b, c
            )
            X = X + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
            Y = Y + (dt / 6.0) * ((Ydot1 + Ydot4) + 2.0 * (Ydot2 + Ydot3))

        xhist[n + 1], yhist[n + 1], time[n + 1], xytend_hist[n + 1] = (
            X,
            Y,
            t0 + si * (n + 1),
            XYtend,
        )
    return xhist, yhist, time, xytend_hist

def s(k, K):
    """A non-dimension coordinate from -1..+1 corresponding to k=0..K"""
    return 2 * k / K - 1


def validate_simulation_data(X_forecast, Y_forecast=None, t_forecast=None, C_forecast=None, ic_num=None, system_type='coupled'):
    """
    시뮬레이션 데이터의 품질을 검증하는 함수

    Args:
        X_forecast: X 변수 예측 데이터
        Y_forecast: Y 변수 예측 데이터 (coupled system에서만 사용)
        t_forecast: 시간 데이터
        C_forecast: Coupling 데이터 (coupled system에서만 사용)
        ic_num: 초기 조건 번호
        system_type: 'coupled' 또는 'single'

    Returns:
        bool: 검증 통과 여부
    """
    if system_type == 'coupled':
        # 1. NaN/Inf 검증
        if (np.any(np.isnan(X_forecast)) or np.any(np.isnan(Y_forecast)) or
            np.any(np.isnan(t_forecast)) or np.any(np.isnan(C_forecast))):
            return False

        if (np.any(np.isinf(X_forecast)) or np.any(np.isinf(Y_forecast)) or
            np.any(np.isinf(t_forecast)) or np.any(np.isinf(C_forecast))):
            return False

        # 2. 극단적인 값 검증 (발산 감지)
        x_max_abs = np.max(np.abs(X_forecast))
        y_max_abs = np.max(np.abs(Y_forecast))
        c_max_abs = np.max(np.abs(C_forecast))

        if x_max_abs > 1000:
            return False

        if y_max_abs > 1000:
            return False

        if c_max_abs > 1000:
            return False

        # 3. 데이터 범위 검증 (물리적으로 의미있는 범위)
        x_range = np.max(X_forecast) - np.min(X_forecast)
        y_range = np.max(Y_forecast) - np.min(Y_forecast)

        if x_range < 0.1:  # X 변수가 거의 변화하지 않음
            return False

        if y_range < 0.01:  # Y 변수가 거의 변화하지 않음
            return False

        # 4. 시간 데이터 검증
        if len(t_forecast) < 100:  # 최소 시간 스텝 수 확인
            return False

        # 5. 데이터 일관성 검증
        if (X_forecast.shape[0] != Y_forecast.shape[0] or
            X_forecast.shape[0] != t_forecast.shape[0] or
            X_forecast.shape[0] != C_forecast.shape[0]):
            return False

        # 6. 통계적 검증 (이상치 감지)
        x_std = np.std(X_forecast)
        y_std = np.std(Y_forecast)

        if x_std < 0.01:  # X 변수의 표준편차가 너무 작음
            return False

        if y_std < 0.001:  # Y 변수의 표준편차가 너무 작음
            return False

        print(f"IC {ic_num}: 검증 통과 ✓ (X_max: {x_max_abs:.2f}, Y_max: {y_max_abs:.2f}, C_max: {c_max_abs:.2f})")
        return True

    else:  # single system
        # 1. NaN/Inf 검증
        if np.any(np.isnan(X_forecast)) or np.any(np.isnan(t_forecast)):
            print(f"IC {ic_num}: NaN 값이 발견되어 검증 실패")
            return False

        if np.any(np.isinf(X_forecast)) or np.any(np.isinf(t_forecast)):
            print(f"IC {ic_num}: Inf 값이 발견되어 검증 실패")
            return False

        # 2. 극단적인 값 검증 (발산 감지)
        x_max_abs = np.max(np.abs(X_forecast))

        if x_max_abs > 1000:
            print(f"IC {ic_num}: X 변수가 발산 (최대 절댓값: {x_max_abs:.2f})")
            return False

        # 3. 데이터 범위 검증 (물리적으로 의미있는 범위)
        x_range = np.max(X_forecast) - np.min(X_forecast)

        if x_range < 0.1:  # X 변수가 거의 변화하지 않음
            print(f"IC {ic_num}: X 변수의 변화가 너무 작음 (범위: {x_range:.6f})")
            return False

        # 4. 시간 데이터 검증
        if len(t_forecast) < 100:  # 최소 시간 스텝 수 확인
            print(f"IC {ic_num}: 시간 스텝이 너무 적음 ({len(t_forecast)})")
            return False

        # 5. 데이터 일관성 검증
        if X_forecast.shape[0] != t_forecast.shape[0]:
            print(f"IC {ic_num}: 데이터 차원이 일치하지 않음")
            return False

        # 6. 통계적 검증 (이상치 감지)
        x_std = np.std(X_forecast)

        if x_std < 0.01:  # X 변수의 표준편차가 너무 작음
            print(f"IC {ic_num}: X 변수의 표준편차가 너무 작음 ({x_std:.6f})")
            return False

        print(f"IC {ic_num}: 검증 통과 ✓ (X_max: {x_max_abs:.2f})")
        return True


# Class for convenience
class L96s:
    """
    Class for single time-scale Lorenz 1996 model
    """

    X = None  # Current X state or initial conditions
    F = None  # Forcing
    dt = None  # Default time-step
    method = None  # Integration method

    def __init__(self, K, dt, F=18, method=EulerFwd, t=0):
        """Construct a single time-scale model with parameters:
        K      : Number of X values
        dt     : time-step
        F      : Forcing term
        t      : Initial time
        method : Integration method, e.g. EulerFwd, RK2, or RK4
        """
        self.F, self.dt = F, dt
        self.method = method
        self.X, self.t = F * np.random.randn(K), t
        self.K = self.X.size  # For convenience
        self.k = np.arange(self.K)  # For plotting

    def __repr__(self):
        return (
            "L96: "
            + "K="
            + str(self.K)
            + " F="
            + str(self.F)
            + " dt="
            + str(self.dt)
            + " method="
            + str(self.method)
        )

    def __str__(self):
        return self.__repr__() + "\n X=" + str(self.X) + "\n t=" + str(self.t)

    def copy(self):
        copy = L96s(self.K, self.dt, F=self.F, method=self.method)
        copy.set_state(self.X, t=self.t)
        return copy

    def print(self):
        print(self)

    def set_param(self, dt=None, F=None, t=None, method=None):
        """Set a model parameter, e.g. .set_param(dt=0.002)"""
        if dt is not None:
            self.dt = dt
        if F is not None:
            self.F = F
        if t is not None:
            self.t = t
        if method is not None:
            self.method = method
        return self

    def set_state(self, X, t=None):
        """Set initial conditions (or current state), e.g. .set_state(X)"""
        self.X = X
        self.K = self.X.size  # For convenience
        self.k = np.arange(self.K)  # For plotting
        if t is not None:
            self.t = t
        return self

    def randomize_IC(self):
        """Randomize the initial conditions (or current state)"""
        self.X = self.F * np.random.rand(self.X.size)
        return self

    def run(self, T, store=False):
        """Run model for a total time of T.
        If store=Ture, then stores the final state as the initial conditions for the next segment.
        Returns full history: X[:,:],t[:]."""
        nt = int(T / self.dt)
        X, t = integrate_L96_1t(
            self.X, self.F, self.dt, nt, method=self.method, t0=self.t
        )
        if store:
            self.X, self.t = X[-1], t[-1]
        return X, t


class L96:
    """
    Class for two time-scale Lorenz 1996 model
    """

    X = "Current X state or initial conditions"
    Y = "Current Y state or initial conditions"
    F = "Forcing"
    h = "Coupling coefficient"
    b = "Ratio of timescales"
    c = "Ratio of amplitudes"
    dt = "Time step"

    def __init__(self, K, J, F=18, h=1, b=10, c=10, t=0, dt=0.001):
        """Construct a two time-scale model with parameters:
        K  : Number of X values
        J  : Number of Y values per X value
        F  : Forcing term (default 18.)
        h  : coupling coefficient (default 1.)
        b  : ratio of amplitudes (default 10.)
        c  : time-scale ratio (default 10.)
        t  : Initial time (default 0.)
        dt : Time step (default 0.001)
        """
        self.F, self.h, self.b, self.c, self.dt = F, h, b, c, dt
        self.X, self.Y, self.t = b * np.random.randn(K), np.random.randn(J * K), t
        self.K, self.J, self.JK = K, J, J * K  # For convenience
        self.k, self.j = np.arange(self.K), np.arange(self.JK)  # For plotting

    def __repr__(self):
        return (
            "L96: "
            + "K="
            + str(self.K)
            + " J="
            + str(self.J)
            + " F="
            + str(self.F)
            + " h="
            + str(self.h)
            + " b="
            + str(self.b)
            + " c="
            + str(self.c)
            + " dt="
            + str(self.dt)
        )

    def __str__(self):
        return (
            self.__repr__()
            + "\n X="
            + str(self.X)
            + "\n Y="
            + str(self.Y)
            + "\n t="
            + str(self.t)
        )

    def copy(self):
        copy = L96(self.K, self.J, F=self.F, h=self.h, b=self.b, c=self.c, dt=self.dt)
        copy.set_state(self.X, self.Y, t=self.t)
        return copy

    def print(self):
        print(self)

    def set_param(self, dt=None, F=None, h=None, b=None, c=None, t=None):
        """Set a model parameter, e.g. .set_param(si=0.01, dt=0.002)"""
        if dt is not None:
            self.dt = dt
        if F is not None:
            self.F = F
        if h is not None:
            self.h = h
        if b is not None:
            self.b = b
        if c is not None:
            self.c = c
        if t is not None:
            self.t = t
        return self

    def set_state(self, X, Y, t=None):
        """Set initial conditions (or current state), e.g. .set_state(X,Y)"""
        self.X, self.Y = X, Y
        if t is not None:
            self.t = t
        self.K, self.JK = self.X.size, self.Y.size  # For convenience
        self.J = self.JK // self.K
        self.k, self.j = np.arange(self.K), np.arange(self.JK)  # For plotting
        return self

    def randomize_IC(self):
        """Randomize the initial conditions (or current state)"""
        X, Y = self.b * np.random.rand(self.X.size), np.random.rand(self.Y.size)
        return self.set_state(X, Y)

    def run(self, si, T, store=False, return_coupling=False):
        """Run model for a total time of T, sampling at intervals of si.
        If store=Ture, then stores the final state as the initial conditions for the next segment.
        If return_coupling=True, returns C in addition to X,Y,t.
        Returns sampled history: X[:,:],Y[:,:],t[:],C[:,:]."""
        nt = int(T / si)
        X, Y, t, C = integrate_L96_2t_with_coupling(
            self.X,
            self.Y,
            si,
            nt,
            self.F,
            self.h,
            self.b,
            self.c,
            t0=self.t,
            dt=self.dt,
        )
        if store:
            self.X, self.Y, self.t = X[-1], Y[-1], t[-1]
        if return_coupling:
            return X, Y, t, C
        else:
            return X, Y, t


if __name__ == "__main__":
    K = 8 # Number of globa-scale variables X
    J = 32 # Number of local-scale Y variables per single global-scale X variable
    F = 15.0 # Focring
    b = 10.0 # ratio of amplitudes
    c = 10.0 # time-scale ratio
    h = 1.0 # Coupling coefficient
    noise = 0.03

    si = 0.005  # Sampling time interval
    dt = 0.005  # Time step

    print("\n=== Running Main Experiment ===")

    num_ic = 300  # 초기 조건 개수
    ic_interval = 10  # 적분의 마지막 포인트 이후 다음 적분을 시작할 초기 조건까지의 간격 (MTU)
    forecast_time = 10  # # Hist_Deterministic.py의 nt = 각 초기 조건 별 적분 기간 (MTU) (10 / 0.005 = 20000)

    # 데이터 저장을 위한 배열 초기화
    # 메모리 효율성을 위해 결과를 점진적으로 저장
    time_steps_per_ic = int(forecast_time / si) + 1
    all_X = np.zeros((num_ic, time_steps_per_ic, K))
    all_Y = np.zeros((num_ic, time_steps_per_ic, J * K))
    all_t = np.zeros((num_ic, time_steps_per_ic))
    all_C = np.zeros((num_ic, time_steps_per_ic, K))


    import time
    start_time = time.time()
    k = np.arange(K)
    j = np.arange(J * K)

    Xinit = s(k, K) * (s(k, K) - 1) * (s(k, K) + 1)
    Yinit = 0 * s(j, J * K) * (s(j, J * K) - 1) * (s(j, J * K) + 1)

    # 모델 초기화
    coupled_system = L96(K, J, F=F, h=h, b=b, c=c, dt=dt)
    coupled_system.set_state(Xinit, Yinit)
    spinup_time = 3 # Hist_Deterministic.py의 nt_pre = (10 / 0.005 = 2000)
    count_ic = 0
    while count_ic < num_ic:
        # 3 MTU 동안 스핀업 실행 및 spinup 상태 저장
        X_spinup, Y_spinup, t_spinup, C_spinup = coupled_system.run(si, spinup_time, store=True, return_coupling=True)
        # 이후 10 MTU 동안 적분 후 마지막 상태 저장
        X_forecast, Y_forecast, t_forecast, C_forecast = coupled_system.run(si, forecast_time, store=True, return_coupling=True)
        # 데이터 검증 수행
        if not validate_simulation_data(X_forecast, Y_forecast, t_forecast, C_forecast, count_ic + 1, system_type='coupled'):
            coupled_system.randomize_IC()
            continue

        # 검증 통과 시 데이터 저장
        print(f"IC {count_ic + 1}: 검증 통과, Coupling System 데이터 저장 중...")

        # 적분 결과 저장
        all_X[count_ic] = X_forecast
        all_Y[count_ic] = Y_forecast
        all_t[count_ic] = t_forecast
        all_C[count_ic] = C_forecast
        count_ic += 1
        coupled_system.randomize_IC()


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n모든 Coupling System 초기 조건 처리 완료! 소요 시간: {elapsed_time:.2f}초")

    # 데이터를 50개 초기 조건씩 분할하여 저장
    results_dir = os.path.join(os.getcwd(), "simulated_data")
    os.makedirs(results_dir, exist_ok=True)

    batch_size = 1
    num_batches = (num_ic + batch_size - 1) // batch_size  # 올림 나눗셈

    print(f"\n총 {num_ic}개의 검증된 데이터를 {num_batches}개 배치로 저장합니다...")

    for batch in tqdm(range(num_batches), desc="Coupling System 데이터 저장 중"):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_ic)

        batch_X = all_X[start_idx:end_idx]
        batch_Y = all_Y[start_idx:end_idx]
        batch_t = all_t[start_idx:end_idx]
        batch_C = all_C[start_idx:end_idx]

        np.save(f"{results_dir}/X_batch_coupled_{batch+1}.npy", batch_X)
        np.save(f"{results_dir}/Y_batch_coupled_{batch+1}.npy", batch_Y)
        np.save(f"{results_dir}/t_batch_coupled_{batch+1}.npy", batch_t)
        np.save(f"{results_dir}/C_batch_coupled_{batch+1}.npy", batch_C)



    print(f"\n총 {num_ic}개의 검증된 single system 데이터 생성 시작...")

    start_time = time.time()
    single_system = L96s(K, dt, F=F, method=RK4)
    single_system.set_state(Xinit)
    all_X_single = np.zeros((num_ic, time_steps_per_ic, K))
    all_t_single = np.zeros((num_ic, time_steps_per_ic))

    count_ic = 0
    while count_ic < num_ic:
        # 3 MTU 동안 스핀업 실행 및 spinup 상태 저장
        X_spinup_single, t_spinup_single = single_system.run(spinup_time, store=True)
        # 이후 10 MTU 동안 적분 후 마지막 상태 저장
        X_forecast_single, t_forecast_single = single_system.run(forecast_time, store=True)
        # 데이터 검증 수행
        if not validate_simulation_data(X_forecast_single, t_forecast=t_forecast_single, ic_num=count_ic + 1, system_type='single'):
            single_system.randomize_IC()
            continue

        # 검증 통과 시 데이터 저장
        print(f"IC {count_ic + 1}: 검증 통과, Single System 데이터 저장 중...")

        all_X_single[count_ic] = X_forecast_single
        all_t_single[count_ic] = t_forecast_single
        count_ic += 1
        single_system.randomize_IC()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n모든 Single System 초기 조건 처리 완료! 소요 시간: {elapsed_time:.2f}초")

    # Single System 데이터 저장
    print(f"\n총 {num_ic}개의 검증된 Single System 데이터를 {num_batches}개 배치로 저장합니다...")

    for batch in tqdm(range(num_batches), desc="Single System 데이터 저장 중"):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, num_ic)

        # 저장 전 최종 검증
        batch_X_single = all_X_single[start_idx:end_idx]
        batch_t_single = all_t_single[start_idx:end_idx]

        # 각 배치의 데이터 저장
        np.save(f"{results_dir}/X_batch_single_{batch+1}.npy", batch_X_single)
        np.save(f"{results_dir}/t_batch_single_{batch+1}.npy", batch_t_single)


    # 메타데이터 저장
    metadata = {
        'K': K,
        'J': J,
        'F': F,
        'h': h,
        'b': b,
        'c': c,
        'dt': dt,
        'si': si,
        'spinup_time': spinup_time,
        'forecast_time': forecast_time,
        'ic_interval': ic_interval,
        'num_ic': num_ic,
        'time_steps_per_ic': time_steps_per_ic,
        'num_batches': num_batches,
        'batch_size': batch_size
    }
    import json
    with open(f"{results_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)

    print(f"\n=== 데이터 생성 및 저장 완료 ===")
    print(f"저장 위치: {results_dir}")
    print(f"총 {num_ic}개의 검증된 초기 조건")
    print(f"총 {num_batches}개의 배치 파일")
    print(f"모든 데이터가 품질 검증을 통과하여 저장되었습니다. ✓")