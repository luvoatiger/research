""" Lorenz-96 model
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18.
https://www.ecmwf.int/en/elibrary/10829-predictability-problem-partly-solved
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit
from tqdm import tqdm
import seaborn as sns


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
    return 2 * (0.5 + k) / K - 1


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
            print(f"\nt : {self.t, t[-1]}")
            self.X, self.Y, self.t = X[-1], Y[-1], t[-1]
        if return_coupling:
            return X, Y, t, C
        else:
            return X, Y, t

    
if __name__ == "__main__":
    # Kim's experimental setup
    K = 36  # Number of globa-scale variables X
    J = 10  # Number of local-scale Y variables per single global-scale X variable
    F = 20  # Forcing
    h = 1.0  # Coupling coefficient
    b = 10    # Ratio of amplitudes

    si, dt = 0.005, 0.005  # Sampling time interval
    
    c = 10    # time-scale ratio 설정

    print("\n=== Running Main Experiment ===")

    num_ic = 20  # 초기 조건 개수
    ic_interval = 10  # 적분의 마지막 포인트 이후 다음 적분을 시작할 초기 조건까지의 간격 (MTU)
    forecast_time = 10  # 각 초기 조건 별 적분 기간 (MTU)
    
    # 결과 저장을 위한 딕셔너리
    results = {
        'X': [],  # 각 초기 조건에서의 X 변수 시계열
        'Y': [],  # 각 초기 조건에서의 Y 변수 시계열
        't': [],  # 각 초기 조건에서의 시간
        'ic_X': [],  # 각 초기 조건의 X 상태
        'ic_Y': []   # 각 초기 조건의 Y 상태
    }

    # 모델 초기화
    model = L96(K, J, F=F, h=h, b=b, c=c, dt=dt)
    model.set_state(s(model.k, model.K) * (s(model.k, model.K) - 1) * (s(model.k, model.K) + 1), 0 * model.j)    
    spinup_time = 3

    for i in tqdm(range(num_ic), desc="초기 조건 처리 중"):        
        # 3 MTU 동안 스핀업 실행
        X_spinup, Y_spinup, t_spinup = model.run(si, spinup_time, store=True)
        print("\n스핀업 후 모델 상태:")
        print(model)

        # t = 0으로 초기화 한 후, 10 MTU 동안 적분 후 마지막 상태 저장
        model.t = 0
        X_forecast, Y_forecast, t_forecast = model.run(si, forecast_time, store=True)
        
        # 적분 결과 저장
        results['X'].append(X_forecast)
        results['Y'].append(Y_forecast)
        results['t'].append(t_forecast)  

        # 다음 적분으로 넘어가기 위해 10 MTU 동안 적분 후 마지막 상태 저장. 해당 상태가 다음 적분의 초기값
        model.randomize_IC()
        #_, _, _ = model.run(si, ic_interval, store=True)

    print("\n모든 초기 조건 처리 완료!")
    '''
        # 모델 초기화
    model = L96(K, J, F=F, h=h, b=b, c=c, dt=dt)
    model.set_state(s(model.k, model.K) * (s(model.k, model.K) - 1) * (s(model.k, model.K) + 1), 0 * model.j)    

    # 3 MTU 동안 스핀업 실행
    spinup_time = 3
    X_spinup, Y_spinup, t_spinup = model.run(si, spinup_time, store=True)
    print("\n스핀업 후 모델 상태:")
    print(model)

    for i in tqdm(range(num_ic), desc="초기 조건 처리 중"):        
        # 10 MTU 동안 적분 후 마지막 상태 저장
        X_forecast, Y_forecast, t_forecast = model.run(si, forecast_time, store=True)
        
        # 적분 결과 저장
        results['X'].append(X_forecast)
        results['Y'].append(Y_forecast)
        results['t'].append(t_forecast)  

        # 다음 적분으로 넘어가기 위해 10 MTU 동안 적분 후 마지막 상태 저장. 해당 상태가 다음 적분의 초기값
        _, _, _ = model.run(si, ic_interval, store=True)

    print("\n모든 초기 조건 처리 완료!")
    '''
    # X1 변수의 상관 행렬 계산
    # 각 초기 조건에서의 X1 변수 시계열 추출
    X1_timeseries = []
    for i in range(num_ic):
        X1_timeseries.append(results['X'][i][:, 0])  # X1 변수는 인덱스 0

    # X1 변수의 상관 행렬 계산
    X1_array = np.array(X1_timeseries)
    X1_corr_matrix = np.corrcoef(X1_array)

    # X1 변수의 상관 행렬 출력
    print("\nX1 변수의 상관 행렬:")
    print(X1_corr_matrix)

    # X1 변수의 상관 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(X1_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=range(1, num_ic+1), yticklabels=range(1, num_ic+1))
    plt.title("X1 Variable Correlation Matrix")
    plt.xlabel("Initial Condition")
    plt.ylabel("Initial Condition")
    plt.tight_layout()
    plt.show()

    # ===== X1 변수의 상관계수 히스토그램과 밀도 플롯을 함께 표시 =====
    # 대각선 요소(자기 자신과의 상관계수 = 1)를 제외한 상관계수 값 추출
    mask = ~np.eye(X1_corr_matrix.shape[0], dtype=bool)
    X1_corr_values = X1_corr_matrix[mask]

    plt.figure(figsize=(10, 6))

    # 히스토그램과 KDE 플롯을 함께 표시
    sns.histplot(X1_corr_values, bins=20, kde=True, color='blue', stat='density')
    plt.title("X1 Variable Correlation Coefficient Distribution")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Density")
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True)

    # 통계 정보 추가
    mean_corr = np.mean(X1_corr_values)
    median_corr = np.median(X1_corr_values)
    std_corr = np.std(X1_corr_values)

    textstr = f'Mean: {mean_corr:.4f}\nMedian: {median_corr:.4f}\nStd Dev: {std_corr:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

    # 모든 X 변수의 상관 행렬 계산
    corr_matrices = []

    # 각 X 변수(X1, X2, ..., X36)에 대해 상관 행렬 계산
    for k in range(K):
        # 각 초기 조건에서의 X[k] 변수 시계열 추출
        X_k_timeseries = []
        for i in range(num_ic):
            X_k_timeseries.append(results['X'][i][:, k])
        
        # X[k] 변수의 상관 행렬 계산
        X_k_array = np.array(X_k_timeseries)
        corr_matrix = np.corrcoef(X_k_array)
        corr_matrices.append(corr_matrix)

    # 모든 X 변수의 평균 상관 행렬 계산
    avg_corr_matrix = np.mean(corr_matrices, axis=0)

    # 평균 상관 행렬 시각화
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=range(1, num_ic+1), yticklabels=range(1, num_ic+1))
    plt.title("Average Correlation Matrix (All X Variables)")
    plt.xlabel("Initial Condition")
    plt.ylabel("Initial Condition")
    plt.tight_layout()
    plt.show()

    # ===== 모든 X 변수의 상관계수 분포 비교 =====
    plt.figure(figsize=(15, 10))

    # 모든 X 변수의 상관계수 값 수집
    all_corr_values = []
    corr_means = []

    for k in range(K):
        # 대각선 요소 제외
        mask = ~np.eye(corr_matrices[k].shape[0], dtype=bool)
        corr_values = corr_matrices[k][mask]
        all_corr_values.append(corr_values)
        corr_means.append(np.mean(corr_values))

    # 서브플롯 1: 모든 X 변수의 상관계수 분포 (박스 플롯)
    plt.subplot(2, 2, 1)
    plt.boxplot(all_corr_values, labels=[f'X{k+1}' for k in range(K)])
    plt.title("Correlation Coefficient Distribution for All X Variables (Box Plot)")
    plt.xlabel("X Variable")
    plt.ylabel("Correlation Coefficient")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(rotation=90)
    plt.grid(True)

    # 서브플롯 2: 모든 X 변수의 평균 상관계수
    plt.subplot(2, 2, 2)
    plt.bar(range(1, K+1), corr_means)
    plt.title("Mean Correlation Coefficient by X Variable")
    plt.xlabel("X Variable")
    plt.ylabel("Mean Correlation")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True)

    # 서브플롯 3: 모든 X 변수의 상관계수 밀도 플롯
    plt.subplot(2, 2, 3)
    for k in range(0, K, 6):  # 가독성을 위해 6개 간격으로 표시
        sns.kdeplot(all_corr_values[k], label=f'X{k+1}')
    plt.title("Correlation Coefficient Density Plot for Selected X Variables")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Density")
    plt.axvline(x=0, color='r', linestyle='--')
    plt.legend()
    plt.grid(True)

    # 서브플롯 4: 모든 X 변수의 상관계수 히스토그램과 KDE 플롯 (누적)
    plt.subplot(2, 2, 4)
    all_corrs_flat = np.concatenate(all_corr_values)
    sns.histplot(all_corrs_flat, bins=30, kde=True, color='purple', stat='density')
    plt.title("Distribution of All Correlation Coefficients (All X Variables)")
    plt.xlabel("Correlation Coefficient")
    plt.ylabel("Density")
    plt.axvline(x=0, color='r', linestyle='--')
    plt.grid(True)

    # 통계 정보 추가
    mean_all = np.mean(all_corrs_flat)
    median_all = np.median(all_corrs_flat)
    std_all = np.std(all_corrs_flat)

    textstr = f'Mean: {mean_all:.4f}\nMedian: {median_all:.4f}\nStd Dev: {std_all:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()

    print("\n상관계수 분석 완료!")
    print(f"X1 변수의 상관계수 히스토그램과 KDE 플롯이 생성되었습니다.")
    print(f"모든 X 변수의 상관계수 분포가 분석되었습니다.")
    print(f"\n총 {num_ic}개의 초기 조건에서 각각 {forecast_time} MTU 동안의 데이터가 생성되었습니다.")
    print(f"각 초기 조건은 이전 상태에서 {ic_interval} MTU 추가 적분 후의 상태입니다.")
    print(f"모든 초기 조건에서 시간 범위는 t = [0, {forecast_time}] MTU입니다.")