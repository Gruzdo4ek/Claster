from sklearn.metrics import silhouette_score, davies_bouldin_score, v_measure_score


class ClusteringMetrics:
    def __init__(self, data, labels, ground_truth=None):
        self.data = data
        self.labels = labels
        self.ground_truth = ground_truth

    def compute_metrics(self):
        # Расчет инерции
        inertia = silhouette_score(self.data, self.labels)

        # Индекс силуэта
        silhouette_avg = silhouette_score(self.data, self.labels)

        # Индекс Дависа-Болдуина
        davies_bouldin = davies_bouldin_score(self.data, self.labels)

        # V-мерa
        v_measure = v_measure_score(self.ground_truth, self.labels) if self.ground_truth else None

        return {
            "inertia": inertia,
            "silhouette_score": silhouette_avg,
            "davies_bouldin": davies_bouldin,
            "v_measure": v_measure
        }
