import tkinter as tk
from tkinter import messagebox
import os
import sys
import subprocess
import threading
import importlib

def check_dependencies():
    required = ["numpy", "matplotlib", "scipy", "mplcursors"]
    missing = []
    for lib in required:
        try:
            importlib.import_module(lib)
        except ImportError:
            missing.append(lib)
    
    if missing:
        msg = f"Отсутствуют необходимые библиотеки: {', '.join(missing)}\n\n" \
              f"Пожалуйста, запустите 'install_deps.bat' или выполните:\n" \
              f"pip install -r requirements.txt"
        messagebox.showerror("Ошибка зависимостей", msg)
        return False
    return True

def run_lab(lab_name):
    script_path = os.path.join(os.path.dirname(__file__), "labs", f"{lab_name}.py")
    variant = variant_entry.get()
    
    if not variant.isdigit():
        messagebox.showwarning("Внимание", "Введите номер варианта!")
        return

    # Скрываем главное меню
    root.withdraw()

    def monitor_process():
        # Запускаем и ЖДЕМ
        process = subprocess.Popen([sys.executable, script_path, "--variant", variant])
        process.wait()
        # Возвращаем меню в основном потоке через .after
        root.after(0, root.deiconify)

    # Запускаем ожидание в отдельном потоке, чтобы GUI не вис
    threading.Thread(target=monitor_process, daemon=True).start()

root = tk.Tk()
root.title("DSP Station")
root.geometry("400x350")
root.configure(bg="#2c3e50")

# Проверка зависимостей при старте
if not check_dependencies():
    # Если библиотек нет, даем пользователю шанс всё же продолжить (на свой страх и риск) 
    # или просто закрываемся. Здесь просто вывели ошибку, но дадим зайти в меню.
    pass

tk.Label(root, text="Универсальное меню ЦОСиИ", font=("Arial", 14, "bold"), fg="white", bg="#2c3e50", pady=20).pack()

frame_v = tk.Frame(root, bg="#34495e", padx=10, pady=10)
frame_v.pack(pady=10)
tk.Label(frame_v, text="ВАРИАНТ:", font=("Arial", 12, "bold"), fg="#f1c40f", bg="#34495e").pack(side="left")
variant_entry = tk.Entry(frame_v, font=("Arial", 12), width=5, justify="center")
variant_entry.insert(0, "10")
variant_entry.pack(side="left", padx=10)

btn_style = {"font": ("Arial", 11, "bold"), "width": 30, "height": 2, "cursor": "hand2"}

tk.Button(root, text="ЛАБОРАТОРНАЯ №1", command=lambda: run_lab("lab1_instruments"), bg="#3498db", fg="white", **btn_style).pack(pady=10)
tk.Button(root, text="ЛАБОРАТОРНАЯ №2", command=lambda: run_lab("lab2_filters"), bg="#27ae60", fg="white", **btn_style).pack(pady=10)

root.mainloop()
