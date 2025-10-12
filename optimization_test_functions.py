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

        plt.contour(x_grid, y_grid, z_grid, levels=20, cmap='viridis')
        plt.colorbar(label='Distance')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

    def plot_center_path(self):
        pass