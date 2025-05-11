import tkinter as tk
from tkinter import Toplevel, Label, Button, IntVar, Checkbutton, messagebox


class FeatureSelector:
    def __init__(self, master, n_features):
        self.master = master
        self.n_features = n_features
        self.selected_features = list(range(n_features))  # по умолчанию все
        self.check_vars = []
        self.k_value = None  # Для хранения количества кластеров
        self.auto_k_var = tk.IntVar(value=0)  # Для чекбокса автоматического подбора

    def select_features(self, callback):
        window = tk.Toplevel(self.master)
        window.title("Выбор признаков и настройка кластеров")

        # Выбор признаков
        tk.Label(window, text="Выберите признаки для кластеризации:").pack(pady=5)

        for i in range(self.n_features):
            var = tk.IntVar(value=1)  # по умолчанию все выбраны
            cb = tk.Checkbutton(window, text=f"Признак {i + 1}", variable=var)
            cb.pack(anchor='w')
            self.check_vars.append(var)

        # Ввод количества кластеров
        tk.Label(window, text="Введите количество кластеров (или выберите автоматический подбор):").pack(pady=10)
        k_entry = tk.Entry(window)
        k_entry.pack(pady=5)

        # Чекбокс для автоматического подбора гиперпараметров
        auto_check = tk.Checkbutton(window, text="Автоматический подбор гиперпараметров", variable=self.auto_k_var)
        auto_check.pack(pady=5)

        def on_confirm():
            # Собираем выбранные признаки
            selected = [i for i, var in enumerate(self.check_vars) if var.get() == 1]
            if not selected:
                messagebox.showwarning("Внимание", "Нужно выбрать хотя бы один признак.")
                return

            # Получаем количество кластеров
            if self.auto_k_var.get() == 1:
                self.k_value = None  # Автоматический подбор
            else:
                try:
                    k_value = int(k_entry.get())
                    if k_value <= 1:
                        raise ValueError()
                    self.k_value = k_value
                except ValueError:
                    messagebox.showerror("Ошибка", "Введите корректное количество кластеров (целое число > 1).")
                    return

            callback(selected, self.k_value)  # Передаем выбранные признаки и количество кластеров
            window.destroy()

        tk.Button(window, text="Подтвердить", command=on_confirm).pack(pady=10)