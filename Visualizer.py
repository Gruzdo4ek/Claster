import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Visualizer:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def visualize_clusters(self, is_3d=False):
        if is_3d:
            return self._visualize_3d()
        else:
            return self._visualize_2d()

    def _visualize_2d(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        # Преобразуем данные в numpy array
        X = self.data.values

        # Проверяем, что есть хотя бы 2 признака
        if X.shape[1] < 2:
            raise ValueError("Для 2D визуализации требуется минимум 2 признака")

        # Берем только первые два признака
        x = X[:, 0]
        y = X[:, 1]

        # Визуализация кластеров
        scatter = ax.scatter(x, y, c=self.labels, cmap='viridis', s=50, alpha=0.6)

        # Настройки графика
        ax.set_title('Визуализация кластеров (2D)')
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
        ax.grid(True)

        # Легенда
        legend = ax.legend(*scatter.legend_elements(),
                          title="Кластеры", loc="upper right")
        ax.add_artist(legend)

        return fig

    def _visualize_3d(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Преобразуем данные в numpy array
        X = self.data.values

        # Проверяем, что есть хотя бы 3 признака
        if X.shape[1] < 3:
            raise ValueError("Для 3D визуализации требуется минимум 3 признака")

        # Берем первые три признака
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]

        # Визуализация кластеров
        scatter = ax.scatter(x, y, z, c=self.labels, cmap='viridis', s=50, alpha=0.6)

        # Настройки графика
        ax.set_title('Визуализация кластеров (3D)')
        ax.set_xlabel('Признак 1')
        ax.set_ylabel('Признак 2')
        ax.set_zlabel('Признак 3')

        # Легенда
        legend = ax.legend(*scatter.legend)