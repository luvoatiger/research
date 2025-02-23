import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class LorenzSystem(ABC):
    def __init__(self, F=8, sigma=10, beta=8/3, rho=28, solver='RK4'):
        self.F = F
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.solver = solver

    @abstractmethod
    def ode_X(self, X):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def plot_state(self):
        pass

    @abstractmethod
    def main(self):
        pass


class Lorenz96System(LorenzSystem):
    def __init__(self, F=8, N = 8, solver='RK4', steps=10000, dt=0.005):
        self.F = F
        self.N = N
        self.solver = solver
        self.steps = steps
        self.dt = dt

        # Initial conditions
        self.initial_conditions = np.random.randn(N)
        self.X = np.zeros((steps, N))
        self.X[0] = self.initial_conditions


    def ode_X(self, X):
        return (np.roll(X, -1) - np.roll(X, 2)) * np.roll(X, 1) - X + self.F


    def step(self, i):
        if self.solver == 'RK4':
            k1 = self.ode_X(self.X[i])
            k2 = self.ode_X(self.X[i] + 0.5 * self.dt * k1)
            k3 = self.ode_X(self.X[i] + 0.5 * self.dt * k2)
            k4 = self.ode_X(self.X[i] + self.dt * k3)
            self.X[i+1] = self.X[i] + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)   
        elif self.solver == 'Euler':
            self.X[i+1] = self.X[i] + self.dt * self.ode_X(self.X[i])


    def plot_state(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot(self.X[:, 0], self.X[:, 1], self.X[:, 2])
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$x_3$")
        plt.show()


    def main(self):
        for i in range(self.steps - 1):
            self.step(i)
        self.plot_state()


class Lorenz63System(LorenzSystem):
    def __init__(self, F=8, sigma=10, beta=8/3, rho=28, solver='RK4'):
        self.F = F
        self.sigma = sigma
        self.beta = beta
        self.rho = rho
        self.solver = solver

    def ode_X(self, X):
        return self.sigma * (X[1] - X[0])

    def step(self):
        pass

    def plot_state(self):
        pass

    def main(self):
        pass

if __name__ == "__main__":
    lorenz96 = Lorenz96System()
    lorenz96.main()