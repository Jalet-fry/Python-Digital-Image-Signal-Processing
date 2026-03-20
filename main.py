import tkinter as tk
from tkinter import messagebox
import os
import sys
import subprocess
import threading
import importlib

# Константа варианта - теперь фиксированная
DEFAULT_VARIANT = "10"

def check_dependencies():
    required = ["numpy", "matplotlib", "scipy", "mplcursors", "sounddevice"]
    missing = []
    for lib in required:
        try:
            importlib.import_module(lib)
        except ImportError:
            missing.append(lib)
    return missing

def run_lab(lab_name):
    script_path = os.path.join(os.path.dirname(__file__), "labs", f"{lab_name}.py")
    
    # Проверка существования файла (для ЛР3, которую скоро создадим)
    if not os.path.exists(script_path):
        messagebox.showinfo("В разработке", f"Файл {lab_name}.py еще не создан. Начинаем реализацию!")
        return

    # Скрываем главное меню
    root.withdraw()

    def monitor_process():
        # Запускаем с фиксированным вариантом 10
        process = subprocess.Popen([sys.executable, script_path, "--variant", DEFAULT_VARIANT])
        process.wait()
        # Возвращаем меню
        root.after(0, root.deiconify)

    threading.Thread(target=monitor_process, daemon=True).start()

root = tk.Tk()
root.title("DSP Station")
root.geometry("400x420")
root.configure(bg="#2c3e50")
root.resizable(False, False)

# Заголовок
tk.Label(root, text="DSP STATION v2.0", font=("Verdana", 16, "bold"), fg="#ecf0f1", bg="#2c3e50", pady=20).pack()
tk.Label(root, text=f"ТЕКУЩИЙ ВАРИАНТ: {DEFAULT_VARIANT}", font=("Arial", 10, "bold"), fg="#f1c40f", bg="#2c3e50").pack()

# Кнопки
btn_style = {"font": ("Arial", 11, "bold"), "width": 30, "height": 2, "cursor": "hand2", "bd": 0}

tk.Button(root, text="ЛАБОРАТОРНАЯ №1\n(Синтез и Фурье)", 
          command=lambda: run_lab("lab1_instruments"), bg="#3498db", fg="white", **btn_style).pack(pady=10)

tk.Button(root, text="ЛАБОРАТОРНАЯ №2\n(Цифровая фильтрация)", 
          command=lambda: run_lab("lab2_filters"), bg="#27ae60", fg="white", **btn_style).pack(pady=10)

tk.Button(root, text="ЛАБОРАТОРНАЯ №3\n(Анализ речи и DeepFilter)", 
          command=lambda: run_lab("lab3_speech"), bg="#e67e22", fg="white", **btn_style).pack(pady=10)

# Проверка зависимостей
missing = check_dependencies()
if missing:
    status_msg = f"⚠ Внимание! Отсутствуют: {', '.join(missing)}"
    status_col = "#e74c3c"
else:
    status_msg = "✔ Все зависимости установлены"
    status_col = "#2ecc71"

tk.Label(root, text=status_msg, font=("Arial", 9), fg=status_col, bg="#2c3e50", pady=10).pack(side="bottom")

root.mainloop()
