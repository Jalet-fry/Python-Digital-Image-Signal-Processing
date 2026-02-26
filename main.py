import tkinter as tk
from tkinter import messagebox
import os
import sys
import subprocess

def run_lab(lab_name):
    script_path = os.path.join(os.path.dirname(__file__), "labs", f"{lab_name}.py")
    if not os.path.exists(script_path):
        messagebox.showerror("Ошибка", f"Файл {lab_name}.py не найден по пути:\n{script_path}")
        return
    
    # Запуск скрипта лабы в отдельном процессе
    subprocess.Popen([sys.executable, script_path])

# Создание окна
root = tk.Tk()
root.title("ЦОСиИ - Лабораторные работы")
root.geometry("400x300")
root.configure(bg="#f0f0f0")

# Заголовок
label = tk.Label(root, text="Выберите лабораторную работу", font=("Arial", 14, "bold"), bg="#f0f0f0", pady=20)
label.pack()

# Кнопки
btn_style = {"font": ("Arial", 11), "width": 30, "height": 2, "pady": 10}

btn1 = tk.Button(root, text="Лабораторная №1: Анализ сигналов", 
                 command=lambda: run_lab("lab1_instruments"), 
                 bg="#e1f5fe", **btn_style)
btn1.pack(pady=10)

btn2 = tk.Button(root, text="Лабораторная №2: Фильтрация", 
                 command=lambda: run_lab("lab2_filters"), 
                 bg="#fff9c4", **btn_style)
btn2.pack(pady=10)

# Подпись
footer = tk.Label(root, text="Вариант №10: Виолончель", font=("Arial", 9, "italic"), bg="#f0f0f0", fg="gray")
footer.pack(side="bottom", pady=10)

root.mainloop()
