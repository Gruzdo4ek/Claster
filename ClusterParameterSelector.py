from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


class ClusterParameterSelector:
    def __init__(self, data):
        self.data = data

    def select_k_clusters(self, max_k=10):
        inertia = []
        silhouette_avg = []

        # Пробуем кластеризацию для разных значений k от 2 до max_k
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.data)
            inertia.append(kmeans.inertia_)
            silhouette_avg.append(silhouette_score(self.data, kmeans.labels_))

        # Определение оптимального количества кластеров
        optimal_k_inertia = inertia.index(min(inertia)) + 2  # Минимальная инерция (локоть)
        optimal_k_silhouette = silhouette_avg.index(max(silhouette_avg)) + 2  # Максимум силуэта

        # Возвращаем два варианта оптимального числа кластеров для сравнительного анализа
        print(f"Optimal number of clusters based on inertia: {optimal_k_inertia}")
        print(f"Optimal number of clusters based on silhouette score: {optimal_k_silhouette}")

        # Выбираем оптимальный k из двух вариантов, если они совпадают, то возвращаем его
        if optimal_k_inertia == optimal_k_silhouette:
            return optimal_k_inertia
        else:
            # Если они различаются, выбираем тот, который дает лучший коэффициент силуэта
            return optimal_k_silhouette
