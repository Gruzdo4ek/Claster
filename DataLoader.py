import pandas as pd


class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.n_objects = None
        self.n_features = None
        self.n_clusters = None  # Количество кластеров (если есть)

    def load_data(self):
        try:
            with open(self.file_path, 'r') as file:
                # Читаем первую строку с метаданными
                first_line = file.readline().strip()
                parts = list(map(int, first_line.split()))

                # Проверяем минимальное количество параметров
                if len(parts) < 2:
                    raise ValueError(
                        "Первая строка должна содержать как минимум 2 числа (количество объектов и признаков)")

                self.n_objects, self.n_features = parts[:2]

                # Если указано количество кластеров (третье число)
                if len(parts) >= 3:
                    self.n_clusters = parts[2]

                # Читаем данные
                data = pd.read_csv(file, sep=r'\s+', header=None)
                data = data.apply(pd.to_numeric, errors='coerce')

                # Проверка на пропущенные значения
                if data.isnull().values.any():
                    raise ValueError("В данных есть недопустимые или пропущенные значения.")

                # Если есть информация о кластерах, отделяем последний столбец как метки
                if self.n_clusters is not None:
                    if data.shape[1] != self.n_features + 1:
                        raise ValueError(
                            f"Ожидается {self.n_features + 1} столбцов (признаки + метки), но получено {data.shape[1]}")

                    # Разделяем данные и метки кластеров
                    self.data = data.iloc[:, :-1]  # Все кроме последнего столбца
                    self.labels = data.iloc[:, -1]  # Последний столбец - метки
                else:
                    if data.shape[1] != self.n_features:
                        raise ValueError(f"Ожидается {self.n_features} столбцов, но получено {data.shape[1]}")
                    self.data = data

                # Проверка соответствия количества объектов
                if self.data.shape[0] != self.n_objects:
                    raise ValueError(f"Ожидается {self.n_objects} объектов, но получено {self.data.shape[0]}")

                return self.n_features, self.n_objects, self.n_clusters, self.data

        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return None, None, None, None
