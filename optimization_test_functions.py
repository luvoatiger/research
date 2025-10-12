from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class TestFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, x):
        pass

    @abstractmethod
    def plot_contour(self):
        pass

    @abstractmethod
    def plot_center_path(self):
        pass


class AckleyFunction(TestFunction):
    def __init__(self, a=20, b=0.2, c=2*np.pi):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c


    def evaluate(self, x):
        z = -self.a * np.exp(-self.b * np.sqrt(np.mean(x**2))) - np.exp(np.mean(np.cos(self.c * x))) + self.a + np.exp(1)

        return z

    def plot_contour(self):
        x_axis = np.linspace(-32.768, 32.768, 100)
        y_axis = np.linspace(-32.768, 32.768, 100)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
 
        # 각 point에 대해 계산
        z_grid = np.zeros_like(x_grid)
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                z_grid[i, j] = self.evaluate(np.array([x_grid[i, j], y_grid[i, j]]))

        plt.contour(x_grid, y_grid, z_grid, levels=100, cmap='viridis')
        plt.colorbar(label='Distance')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

    def plot_center_path(self, initial_point):
        pass


class BoothFunction(TestFunction):
    def __init__(self):
        super().__init__()


    def evaluate(self, x):
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


    def plot_contour(self):
        x_axis = np.linspace(-10, 10, 100)
        y_axis = np.linspace(-10, 10, 100)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
        z_grid = np.zeros_like(x_grid)
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                z_grid[i, j] = self.evaluate(np.array([x_grid[i, j], y_grid[i, j]]))
        plt.contour(x_grid, y_grid, z_grid, levels=100, cmap='viridis')
        plt.colorbar(label='Distance')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

    def plot_center_path(self, initial_point):
        pass


class BraninFunction(TestFunction):
    def __init__(self, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.r = r
        self.s = s
        self.t = t


    def evaluate(self, x):
        return self.a * (x[1] - self.b * x[0]**2 + self.c * x[0] - self.r)**2 + self.s * (1 - self.t) * np.cos(x[0]) + self.s + 1


    def plot_contour(self):
        x_axis = np.linspace(-5, 20, 100)
        y_axis = np.linspace(-5, 20, 100)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
        z_grid = np.zeros_like(x_grid)
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                z_grid[i, j] = self.evaluate(np.array([x_grid[i, j], y_grid[i, j]]))
        plt.contour(x_grid, y_grid, z_grid, levels=100, cmap='viridis')
        plt.colorbar(label='Distance')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()


    def plot_center_path(self, initial_point):
        pass


class MichalewiczFunction(TestFunction):
    def __init__(self, m=10):
        super().__init__()
        self.m = m


    def evaluate(self, x):
        return -sum([np.sin(v) * np.sin(i*v**2/np.pi)**(2*self.m) for i, v in enumerate(x)])


    def plot_contour(self):
        x_axis = np.linspace(0, 4, 50)
        y_axis = np.linspace(0, 4, 50)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
        z_grid = np.zeros_like(x_grid)
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                z_grid[i, j] = self.evaluate(np.array([x_grid[i, j], y_grid[i, j]]))
        plt.contour(x_grid, y_grid, z_grid, levels=100, cmap='viridis')
        plt.colorbar(label='Distance')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
    
    def plot_center_path(self, initial_point):
        pass


class RosenbrockFunction(TestFunction):
    def __init__(self, a=1, b=5):
        super().__init__()
        self.a = a
        self.b = b

    def evaluate(self, x):
        return (self.a - x[0])**2 + self.b * (x[1] - x[0]**2)**2

    def plot_contour(self):
        x_axis = np.linspace(-2, 2, 50)
        y_axis = np.linspace(-2, 2, 50)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
        z_grid = np.zeros_like(x_grid)
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                z_grid[i, j] = self.evaluate(np.array([x_grid[i, j], y_grid[i, j]]))
        plt.contour(x_grid, y_grid, z_grid, levels=100, cmap='viridis')
        plt.colorbar(label='Distance')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
    
    def plot_center_path(self, initial_point):
        pass


class WheelerRidge(TestFunction):
    def __init__(self, a=1.5):
        super().__init__()
        self.a = a

    def evaluate(self, x):
        return -np.exp(-(x[0]*x[1] - self.a)**2 - (x[1] - self.a)**2)

    def plot_contour(self):
        x_axis = np.linspace(-10, 25, 500)
        y_axis = np.linspace(-4, 6, 100)
        x_grid, y_grid = np.meshgrid(x_axis, y_axis)
        z_grid = np.zeros_like(x_grid)
        for i in range(x_grid.shape[0]):
            for j in range(x_grid.shape[1]):
                z_grid[i, j] = self.evaluate(np.array([x_grid[i, j], y_grid[i, j]]))
        plt.contour(x_grid, y_grid, z_grid, levels=100, cmap='viridis')
        plt.colorbar(label='Distance')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
    
    def plot_center_path(self, initial_point):
        pass