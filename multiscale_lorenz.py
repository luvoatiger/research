""" Lorenz-96 model
Lorenz E., 1996. Predictability: a problem partly solved. In
Predictability. Proc 1995. ECMWF Seminar, 1-18.
https://www.ecmwf.int/en/elibrary/10829-predictability-problem-partly-solved
"""

import numpy as np
from numba import jit, njit
import matplotlib.pyplot as plt
import os

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


def RK4_adaptive(fn, dt, X, tol=1e-6, dt_min=1e-6, dt_max=0.1, safety=0.9, *params):
    """
    Calculate the new state X(n+1) for d/dt X = fn(X,t,...) using the adaptive fourth order Runge-Kutta method.
    
    Args:
        fn     : The function returning the time rate of change of model variables X
        dt     : The initial time step
        X      : Values of X variables at the current time, t
        tol    : Error tolerance for step size adjustment
        dt_min : Minimum allowed time step
        dt_max : Maximum allowed time step
        safety : Safety factor for step size adjustment (0 < safety < 1)
        params : All other arguments that should be passed to fn, i.e. fn(X, t, *params)
    
    Returns:
        X at t+dt, actual dt used
    """
    # First try with full step
    Xdot1 = fn(X, *params)
    Xdot2 = fn(X + 0.5 * dt * Xdot1, *params)
    Xdot3 = fn(X + 0.5 * dt * Xdot2, *params)
    Xdot4 = fn(X + dt * Xdot3, *params)
    X_full = X + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
    
    # Now try with two half steps
    dt_half = dt / 2.0
    # First half step
    Xdot1 = fn(X, *params)
    Xdot2 = fn(X + 0.5 * dt_half * Xdot1, *params)
    Xdot3 = fn(X + 0.5 * dt_half * Xdot2, *params)
    Xdot4 = fn(X + dt_half * Xdot3, *params)
    X_half = X + (dt_half / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
    
    # Second half step
    Xdot1 = fn(X_half, *params)
    Xdot2 = fn(X_half + 0.5 * dt_half * Xdot1, *params)
    Xdot3 = fn(X_half + 0.5 * dt_half * Xdot2, *params)
    Xdot4 = fn(X_half + dt_half * Xdot3, *params)
    X_half = X_half + (dt_half / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
    
    # Estimate error
    error = np.max(np.abs(X_full - X_half))
    
    # Adjust step size based on error
    if error > 0:
        dt_optimal = safety * dt * (tol / error) ** 0.2
    else:
        dt_optimal = dt_max
    
    dt_optimal = min(max(dt_optimal, dt_min), dt_max)
    
    # If error is acceptable, return result, otherwise retry with adjusted step
    if error <= tol:
        return X_half, dt  # Return the more accurate result (two half steps)
    else:
        return RK4_adaptive(fn, dt_optimal, X, tol, dt_min, dt_max, safety, *params)


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


def integrate_L96_1t_adaptive(X0, F, dt_init, nt, tol=1e-6, dt_min=1e-6, dt_max=0.1, t0=0):
    """
    Integrates forward-in-time the single time-scale Lorenz 1996 model, using the adaptive RK4 method.
    Returns the full history with variable time steps.
    
    Args:
        X0      : Values of X variables at the current time
        F       : Forcing term
        dt_init : The initial time step
        nt      : Maximum number of forwards steps
        tol     : Error tolerance for step size adjustment
        dt_min  : Minimum allowed time step
        dt_max  : Maximum allowed time step
        t0      : Initial time (defaults to 0)
    
    Returns:
        X[:,:], time[:] : the full history X[n,k] at times t[n]
    
    Example usage:
        X,t = integrate_L96_1t_adaptive(5+5*np.random.rand(8), 18, 0.01, 500)
        plt.plot(t, X);
    """
    
    time, hist = [t0], [X0.copy()]
    X = X0.copy()
    t = t0
    dt = dt_init
    
    for n in range(nt):
        X, dt_used = RK4_adaptive(L96_eq1_xdot, dt, X, tol, dt_min, dt_max, 0.9, F)
        t += dt_used
        hist.append(X.copy())
        time.append(t)
        dt = dt_used  # Use the adapted time step for next iteration
    
    return np.array(hist), np.array(time)


def integrate_L96_2t_adaptive(X0, Y0, dt_init, nt, F, h, b, c, tol=1e-6, dt_min=1e-6, dt_max=0.1, t0=0):
    """
    Integrates forward-in-time the two time-scale Lorenz 1996 model, using the adaptive RK4 method.
    Returns the full history with variable time steps.
    
    Args:
        X0      : Values of X variables at the current time
        Y0      : Values of Y variables at the current time
        dt_init : The initial time step
        nt      : Maximum number of forwards steps
        F       : Forcing term
        h       : coupling coefficient
        b       : ratio of amplitudes
        c       : time-scale ratio
        tol     : Error tolerance for step size adjustment
        dt_min  : Minimum allowed time step
        dt_max  : Maximum allowed time step
        t0      : Initial time (defaults to 0)
    
    Returns:
        X[:,:], Y[:,:], time[:], C[:,:] : the full history X[n,k], Y[n,k] at times t[n], and coupling term
    
    Example usage:
        X,Y,t,C = integrate_L96_2t_adaptive(5+5*np.random.rand(8), np.random.rand(8*4), 0.01, 500, 18, 1, 10, 10)
        plt.plot(t, X);
    """
    
    time, xhist, yhist, xytend_hist = [t0], [X0.copy()], [Y0.copy()], [np.zeros(len(X0))]
    X, Y = X0.copy(), Y0.copy()
    t = t0
    dt = dt_init
    
    for n in range(nt):
        # Adaptive RK4 for the coupled system
        # First try with full step
        Xdot1, Ydot1, XYtend = L96_2t_xdot_ydot(X, Y, F, h, b, c)
        Xdot2, Ydot2, _ = L96_2t_xdot_ydot(X + 0.5 * dt * Xdot1, Y + 0.5 * dt * Ydot1, F, h, b, c)
        Xdot3, Ydot3, _ = L96_2t_xdot_ydot(X + 0.5 * dt * Xdot2, Y + 0.5 * dt * Ydot2, F, h, b, c)
        Xdot4, Ydot4, _ = L96_2t_xdot_ydot(X + dt * Xdot3, Y + dt * Ydot3, F, h, b, c)
        X_full = X + (dt / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
        Y_full = Y + (dt / 6.0) * ((Ydot1 + Ydot4) + 2.0 * (Ydot2 + Ydot3))
        
        # Now try with two half steps
        dt_half = dt / 2.0
        # First half step
        Xdot1, Ydot1, _ = L96_2t_xdot_ydot(X, Y, F, h, b, c)
        Xdot2, Ydot2, _ = L96_2t_xdot_ydot(X + 0.5 * dt_half * Xdot1, Y + 0.5 * dt_half * Ydot1, F, h, b, c)
        Xdot3, Ydot3, _ = L96_2t_xdot_ydot(X + 0.5 * dt_half * Xdot2, Y + 0.5 * dt_half * Ydot2, F, h, b, c)
        Xdot4, Ydot4, _ = L96_2t_xdot_ydot(X + dt_half * Xdot3, Y + dt_half * Ydot3, F, h, b, c)
        X_half = X + (dt_half / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
        Y_half = Y + (dt_half / 6.0) * ((Ydot1 + Ydot4) + 2.0 * (Ydot2 + Ydot3))
        
        # Second half step
        Xdot1, Ydot1, _ = L96_2t_xdot_ydot(X_half, Y_half, F, h, b, c)
        Xdot2, Ydot2, _ = L96_2t_xdot_ydot(X_half + 0.5 * dt_half * Xdot1, Y_half + 0.5 * dt_half * Ydot1, F, h, b, c)
        Xdot3, Ydot3, _ = L96_2t_xdot_ydot(X_half + 0.5 * dt_half * Xdot2, Y_half + 0.5 * dt_half * Ydot2, F, h, b, c)
        Xdot4, Ydot4, _ = L96_2t_xdot_ydot(X_half + dt_half * Xdot3, Y_half + dt_half * Ydot3, F, h, b, c)
        X_half = X_half + (dt_half / 6.0) * ((Xdot1 + Xdot4) + 2.0 * (Xdot2 + Xdot3))
        Y_half = Y_half + (dt_half / 6.0) * ((Ydot1 + Ydot4) + 2.0 * (Ydot2 + Ydot3))
        
        # Estimate error (use maximum of X and Y errors)
        error_X = np.max(np.abs(X_full - X_half))
        error_Y = np.max(np.abs(Y_full - Y_half))
        error = max(error_X, error_Y)
        
        # Adjust step size based on error
        if error > 0:
            dt_optimal = 0.9 * dt * (tol / error) ** 0.2
        else:
            dt_optimal = dt_max
        
        dt_optimal = min(max(dt_optimal, dt_min), dt_max)
        
        # If error is acceptable, accept the step
        if error <= tol:
            X, Y = X_half, Y_half  # Use the more accurate result
            t += dt
            _, _, XYtend = L96_2t_xdot_ydot(X, Y, F, h, b, c)
            xhist.append(X.copy())
            yhist.append(Y.copy())
            time.append(t)
            xytend_hist.append(XYtend)
            dt = dt_optimal  # Use the adapted time step for next iteration
        else:
            # Reject this step and try again with the adjusted step size
            dt = dt_optimal
            continue
    
    return np.array(xhist), np.array(yhist), np.array(time), np.array(xytend_hist)


# Class for convenience
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

    def run_adaptive(self, T, tol=1e-6, dt_min=1e-6, dt_max=0.1, store=False, return_coupling=False):
        """Run model for a total time of T using adaptive time stepping.
        If store=True, then stores the final state as the initial conditions for the next segment.
        If return_coupling=True, returns C in addition to X,Y,t.
        Returns sampled history: X[:,:],Y[:,:],t[:],C[:,:]."""
        nt = int(T / dt_min)  # Maximum number of steps (worst case)
        X, Y, t, C = integrate_L96_2t_adaptive(
            self.X,
            self.Y,
            self.dt,
            nt,
            self.F,
            self.h,
            self.b,
            self.c,
            tol=tol,
            dt_min=dt_min,
            dt_max=dt_max,
            t0=self.t,
        )
        if store:
            self.X, self.Y, self.t = X[-1], Y[-1], t[-1]
        if return_coupling:
            return X, Y, t, C
        else:
            return X, Y, t


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

    def run_adaptive(self, T, tol=1e-6, dt_min=1e-6, dt_max=0.1, store=False):
        """Run model for a total time of T using adaptive time stepping.
        If store=True, then stores the final state as the initial conditions for the next segment.
        Returns full history: X[:,:],t[:]."""
        nt = int(T / dt_min)  # Maximum number of steps (worst case)
        X, t = integrate_L96_1t_adaptive(
            self.X, 
            self.F, 
            self.dt, 
            nt, 
            tol=tol, 
            dt_min=dt_min, 
            dt_max=dt_max, 
            t0=self.t
        )
        if store:
            self.X, self.t = X[-1], t[-1]
        return X, t
    
def s(k, K):
    """A non-dimension coordinate from -1..+1 corresponding to k=0..K"""
    return 2 * (0.5 + k) / K - 1


if __name__ == "__main__":
    # 논문에서 언급된 파라미터 설정
    K = 8       # 저주파수, 대진폭 X 변수의 수
    J = 32      # 고주파수, 소진폭 Y 변수의 수 (각 X당)
    F = 20      # 강제력 항
    h = 1       # 결합 계수
    b = 10      # 진폭 비율
    c_values = [4, 10]  # 시간 스케일 비율 (두 가지 경우)
    
    # 논문에서 언급된 대로 적응형 RK4 방법 사용
    dt_init = 0.001  # 초기 시간 간격
    dt_max = 0.001   # 최대 시간 간격 (논문에서 언급)
    
    # 결과 저장을 위한 디렉토리 생성
    if not os.path.exists("results"):
        os.makedirs("results")
    
    for c in c_values:
        print(f"시간 스케일 비율 c = {c} 에 대한 실험 시작")
        
        # 모델 초기화
        model = L96(K, J, F=F, h=h, b=b, c=c, dt=dt_init)
        
        # 초기 과도 상태(transient) 제거
        print("과도 상태 제거 중...")
        model.run(0.01, 100, store=True)  # 샘플링 간격 0.01, 총 시간 100 MTU
        
        # 300개의 초기 조건 생성 (10 MTU 간격으로)
        print("300개의 초기 조건 생성 중...")
        initial_conditions = []
        
        for i in range(300):
            # 10 MTU 간격으로 실행 (논문에서 언급된 대로)
            X, Y, t = model.run(0.01, 10, store=True)  # 샘플링 간격 0.01, 총 시간 10 MTU
            # 마지막 상태를 초기 조건으로 저장
            initial_conditions.append((X[-1].copy(), Y[-1].copy()))
            
            if (i+1) % 50 == 0:
                print(f"  {i+1}/300 초기 조건 생성 완료")
        
        # 각 초기 조건에서 "진실" 실행
        print("각 초기 조건에서 진실 실행 중...")
        
        # TODO change for-loop to multiprocess
        # 예시로 첫 5개 초기 조건에 대해서만 그래프 생성
        for i in range(5):
            X0, Y0 = initial_conditions[i]
            
            # 모델 초기화
            model.set_state(X0, Y0)
            
            # 적응형 시간 간격으로 실행
            X, Y, t, C = model.run_adaptive(5, tol=1e-6, dt_min=1e-6, dt_max=dt_max, store=False, return_coupling=True)
            
            # 결과 저장
            np.save(f"results/X_c{c}_ic{i}.npy", X)
            np.save(f"results/Y_c{c}_ic{i}.npy", Y)
            np.save(f"results/t_c{c}_ic{i}.npy", t)
            np.save(f"results/C_c{c}_ic{i}.npy", C)
            
            # 그래프 생성
            plt.figure(figsize=(12, 8))
            
            # X 변수 그래프
            plt.subplot(2, 1, 1)
            for k in range(K):
                plt.plot(t, X[:, k], label=f'X{k+1}' if k < 3 else "")
            plt.title(f'Lorenz 96 모델 (c={c}, 초기 조건 #{i+1})')
            plt.ylabel('X 값')
            if k < 3:
                plt.legend()
            
            # Y 변수 그래프 (첫 번째 X에 연결된 Y만 표시)
            plt.subplot(2, 1, 2)
            for j in range(min(5, J)):  # 처음 5개 Y만 표시
                plt.plot(t, Y[:, j], label=f'Y{j+1}' if j < 3 else "")
            plt.xlabel('시간 (MTU)')
            plt.ylabel('Y 값')
            if j < 3:
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(f"results/lorenz96_c{c}_ic{i}.png", dpi=300)
            plt.close()
            
            if (i+1) % 1 == 0:
                print(f"  초기 조건 #{i+1} 완료")
        
        print(f"c = {c}에 대한 실험 완료\n")
    
    print("모든 실험 완료!")
    
    # 결과 요약
    print("\n결과 요약:")
    print(f"- 생성된 초기 조건 수: 300")
    print(f"- 시간 스케일 비율: {c_values}")
    print(f"- 저장된 파일 위치: ./results/")
    print("- 생성된 파일:")
    print("  * X_c{c}_ic{i}.npy: X 변수 시계열")
    print("  * Y_c{c}_ic{i}.npy: Y 변수 시계열")
    print("  * t_c{c}_ic{i}.npy: 시간 배열")
    print("  * C_c{c}_ic{i}.npy: 결합 항")
    print("  * lorenz96_c{c}_ic{i}.png: 시계열 그래프")