from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


class ClusteredObject:
    def __init__(self, data_point, label):
        self.data_point = data_point  # Исходные данные (объект)
        self.label = label  # Метка кластера

    def __repr__(self):
        return f"ClusteredObject(data_point={self.data_point}, label={self.label})"


class KMeansClustering:
    def __init__(self, data, n_clusters):
        self.data = data
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        # Проверка, что данные в виде DataFrame или NumPy массива
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data.values
        elif isinstance(self.data, np.ndarray):
            pass
        else:
            raise ValueError("Data should be a pandas DataFrame or numpy array")

    def perform_clustering(self):
        try:
            # Проверка на пустые значения
            if np.any(np.isnan(self.data)):
                raise ValueError("Data contains NaN values, please clean the data.")

            # Проверка на корректность размерности
            if len(self.data.shape) != 2:
                raise ValueError("Data must be 2D (samples x features).")

            # Обучение модели KMeans
            self.kmeans.fit(self.data)

            # Получаем метки кластеров и центры кластеров
            labels = self.kmeans.labels_
            centers = self.kmeans.cluster_centers_

            return labels, centers

        except Exception as e:
            print(f"Error during clustering: {e}")
            return None, None
