import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from tkinter import ttk
from ClusterParameterSelector import ClusterParameterSelector
from ClusteringMetrics import ClusteringMetrics
from DataLoader import DataLoader
from KMeansClustering import KMeansClustering
from Visualizer import Visualizer
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class GUI:
    def __init__(self):
        self.loading_window = None
        self.data_window = None
        self.visualize_button = None
        self.n_clusters = None
        self.n_features = None
        self.n_objects = None
        self.auto_tuning_check = None
        self.k_value_entry = None
        self.check_vars = None
        self.auto_tuning_var = None
        self.load_button_with_clusters = None
        self.load_button_no_clusters = None
        self.table_window = None
        self.root = tk.Tk()
        self.root.title("KMeans Кластеризация")

        self.file_path = None
        self.data = None
        self.selected_data = None
        self.k_value = None

        self.create_widgets()

    def create_widgets(self):
        self.center_window(self.root, 400, 200)

        # Создаем фрейм для центрирования кнопок
        button_frame = tk.Frame(self.root)
        button_frame.pack(expand=True)  # Фрейм расширяется, чтобы занять все доступное пространство

        # Меню загрузки данных
        self.load_button_no_clusters = tk.Button(
            button_frame,
            text="Загрузить данные без кластеров",
            command=self.load_data_without_clusters
        )
        self.load_button_no_clusters.pack(pady=5, fill=tk.X, padx=50)  # Кнопка растягивается по X с отступами

        self.load_button_with_clusters = tk.Button(
            button_frame,
            text="Загрузить данные с кластерами",
            command=self.load_data_with_clusters
        )
        self.load_button_with_clusters.pack(pady=5, fill=tk.X, padx=50)  # Кнопка растягивается по X с отступами

    def center_window(self, window, width=800, height=400):
        """Центрирует окно на экране"""
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        position_top = int(screen_height / 2 - height / 2)
        position_right = int(screen_width / 2 - width / 2)
        window.geometry(f'{width}x{height}+{position_right}+{position_top}')

    def load_data_without_clusters(self):
        # Окно выбора файла
        self.file_path = filedialog.askopenfilename(filetypes=[("Текстовые файлы", "*.txt")])
        if self.file_path:
            data_loader = DataLoader(self.file_path)
            self.n_features, self.n_objects, self.n_clusters, data = data_loader.load_data()

            if data is not None:
                messagebox.showinfo("Данные загружены", "Данные успешно загружены!")
                self.data = data
                self.show_data_table()

    def load_data_with_clusters(self):
        # Окно выбора файла с данными, уже имеющими кластеры
        self.file_path = filedialog.askopenfilename(filetypes=[("Текстовые файлы", "*.txt")])
        if self.file_path:
            data_loader = DataLoader(self.file_path)
            self.n_features, self.n_objects, self.n_clusters, data = data_loader.load_data()

            if data is not None:
                if self.n_clusters is not None:
                    messagebox.showinfo("Данные загружены", "Данные с кластерами успешно загружены!")
                    self.data = data
                    self.show_data_table_for_clustering()
                    self.open_results_window()
                else:
                    messagebox.showwarning("Предупреждение", "Файл не содержит информации о кластерах.")
                    self.data = data
                    self.show_data_table()

    def show_data_table(self):
        """Отображает таблицу с данными"""
        self.table_window = Toplevel(self.root)
        self.table_window.title("Таблица данных")
        self.center_window(self.table_window)

        table = ttk.Treeview(self.table_window)
        table['columns'] = list(self.data.columns)

        # Настройка колонок
        for col in self.data.columns:
            table.heading(col, text=col)
            table.column(col, anchor='center')

        # Заполнение данных
        for i, row in self.data.iterrows():
            table.insert("", 'end', values=row.tolist())

        table.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        # Добавление кнопки выбора признаков
        self.open_feature_selection_window()

    def show_data_table_for_clustering(self):
        """Отображает таблицу с данными"""
        self.table_window = Toplevel(self.root)
        self.table_window.title("Таблица данных")
        self.center_window(self.table_window)

        table = ttk.Treeview(self.table_window)
        table['columns'] = list(self.data.columns)

        # Настройка колонок
        for col in self.data.columns:
            table.heading(col, text=col)
            table.column(col, anchor='center')

        # Заполнение данных
        for i, row in self.data.iterrows():
            table.insert("", 'end', values=row.tolist())

        table.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        # Добавление кнопки визуализации
        self.add_visualization_button()

    def add_visualization_button(self):
        # Добавление кнопки для запуска визуализации
        self.visualize_button = tk.Button(self.table_window, text="Запустить визуализацию", command=self.run_visualization)
        self.visualize_button.pack(pady=10)

    def run_visualization(self):
        # Логика для запуска визуализации
        print("Запуск визуализации...")
        self.open_results_window()

    def open_feature_selection_window(self):
        """Окно для выбора признаков и параметров кластеризации"""
        window = Toplevel(self.root)
        window.title("Выбор признаков и параметров KMeans")
        self.center_window(window)

        tk.Label(window, text="Выберите признаки для кластеризации:").pack(pady=5)

        self.check_vars = []
        for i in range(self.data.shape[1]):
            var = tk.IntVar(value=1)  # по умолчанию все выбраны
            cb = tk.Checkbutton(window, text=f"Признак {i + 1} ({self.data.columns[i]})", variable=var)
            cb.pack(anchor='w')
            self.check_vars.append(var)

        tk.Label(window, text="Введите количество кластеров или выберите автонастройку:").pack(pady=5)

        self.k_value_entry = tk.Entry(window)
        self.k_value_entry.pack(pady=5)

        self.auto_tuning_var = tk.IntVar()
        self.auto_tuning_check = tk.Checkbutton(window, text="Автоматический подбор параметров",
                                                variable=self.auto_tuning_var)
        self.auto_tuning_check.pack(pady=5)

        def on_confirm():
            selected_features = [i for i, var in enumerate(self.check_vars) if var.get() == 1]
            if not selected_features:
                messagebox.showwarning("Предупреждение", "Пожалуйста, выберите хотя бы один признак.")
                return

            # Проверяем, что выбрано не менее 2 признаков для визуализации
            if len(selected_features) < 2:
                messagebox.showwarning("Предупреждение", "Для визуализации необходимо выбрать минимум 2 признака.")
                return

            self.selected_data = self.data.iloc[:, selected_features]

            # Если число кластеров не введено, используем авто-настройку
            self.k_value = None
            if self.k_value_entry.get():
                try:
                    self.k_value = int(self.k_value_entry.get())
                    if self.k_value < 1:
                        raise ValueError
                except ValueError:
                    messagebox.showerror("Ошибка", "Некорректное количество кластеров. Введите целое число больше 0.")
                    return

            window.destroy()
            self.run_kmeans()

        tk.Button(window, text="Подтвердить", command=on_confirm).pack(pady=10)

    def show_loading_animation(self):
        """Показывает окно с анимацией загрузки"""
        self.loading_window = Toplevel(self.root)
        self.loading_window.title("Обработка")
        self.center_window(self.loading_window, 800, 100)
        loading_label = tk.Label(self.loading_window, text="Идет процесс кластеризации, пожалуйста, подождите...")
        loading_label.pack(padx=20, pady=20)
        self.loading_window.update()

    def close_loading_animation(self):
        """Закрывает окно с анимацией загрузки"""
        if hasattr(self, 'loading_window') and self.loading_window.winfo_exists():
            self.loading_window.destroy()

    def run_kmeans(self):
        """Запускает кластеризацию"""
        if self.selected_data.empty:
            messagebox.showerror("Ошибка", "Не выбрано ни одного признака.")
            return

        # Показываем окно с анимацией загрузки
        self.show_loading_animation()

        # Подбор количества кластеров
        if self.k_value is None:
            # Если нет значения k, используем авто-настройку
            cluster_selector = ClusterParameterSelector(self.selected_data)
            optimal_k = cluster_selector.select_k_clusters(max_k=50)
        else:
            optimal_k = self.k_value

        # Выполнение кластеризации
        kmeans = KMeansClustering(self.selected_data, optimal_k)
        labels, centers = kmeans.perform_clustering()

        # Закрываем окно с анимацией
        self.close_loading_animation()

        # Вычисление метрик качества
        metrics = ClusteringMetrics(self.selected_data, labels)
        metrics_data = metrics.compute_metrics()

        self.open_results_window(metrics_data, labels)

    def open_results_window(self, metrics_data=None, labels=None):
        """Окно для отображения результатов кластеризации"""
        window = Toplevel(self.root)
        window.title("Результаты кластеризации")
        window.geometry("800x600")
        self.center_window(window)

        try:
            # Фрейм для графика
            graph_frame = tk.Frame(window)
            graph_frame.pack(padx=10, pady=10, side=tk.TOP, fill=tk.BOTH, expand=True)

            # Визуализация графика
            if labels is not None:
                visualizer = Visualizer(self.selected_data, labels)
                try:
                    fig = visualizer.visualize_clusters(is_3d=False)

                    # Создание Canvas для отображения графика в Tkinter
                    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                except ValueError as e:
                    error_label = tk.Label(graph_frame, text=str(e), fg="red")
                    error_label.pack(pady=20)
                    # Показываем только метрики, если не удалось визуализировать

            # Фрейм для таблицы с метриками
            metrics_frame = tk.Frame(window)
            metrics_frame.pack(padx=10, pady=10, side=tk.BOTTOM, fill=tk.X)

            # Таблица с метриками
            table = ttk.Treeview(metrics_frame, columns=["Метрика", "Значение"], show="headings")
            table.heading("Метрика", text="Метрика")
            table.heading("Значение", text="Значение")

            # Заполнение таблицы
            if metrics_data:
                for metric, value in metrics_data.items():
                    table.insert("", "end", values=(metric, value))

            table.pack(padx=10, pady=10, fill=tk.X)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при отображении результатов: {str(e)}")
            window.destroy()


if __name__ == "__main__":
    app = GUI()
    app.root.mainloop()