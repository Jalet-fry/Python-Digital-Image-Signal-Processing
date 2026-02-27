import tkinter as tk
from tkinter import messagebox
import os
import sys
import subprocess

def run_lab(lab_name):
    script_path = os.path.join(os.path.dirname(__file__), "labs", f"{lab_name}.py")
    if not os.path.exists(script_path):
        messagebox.showerror("Ошибка", f"Файл {lab_name}.py не найден!")
        return
    
    variant = variant_entry.get()
    if not variant.isdigit():
        messagebox.showwarning("Внимание", "Введите номер варианта (число)!")
        return

    # Скрываем главное меню, чтобы не плодить окна
    root.withdraw()
    
    # Запускаем лабу и ждем её завершения
    subprocess.run([sys.executable, script_path, "--variant", variant])
    
    # После закрытия лабы возвращаем меню
    root.deiconify()

root = tk.Tk()
root.title("Универсальный DSP Комбайн")
root.geometry("400x350")
root.configure(bg="#2c3e50")

tk.Label(root, text="Система Лабораторных ЦОСиИ", font=("Arial", 14, "bold"), fg="white", bg="#2c3e50", pady=20).pack()

frame_v = tk.Frame(root, bg="#34495e", padx=10, pady=10)
frame_v.pack(pady=10)
tk.Label(frame_v, text="ВАРИАНТ:", font=("Arial", 12, "bold"), fg="#f1c40f", bg="#34495e").pack(side="left")
variant_entry = tk.Entry(frame_v, font=("Arial", 12), width=5, justify="center")
variant_entry.insert(0, "10")
variant_entry.pack(side="left", padx=10)

btn_style = {"font": ("Arial", 11, "bold"), "width": 30, "height": 2, "cursor": "hand2"}

tk.Button(root, text="ЛАБОРАТОРНАЯ №1 (АНАЛИЗ)", 
          command=lambda: run_lab("lab1_instruments"), 
          bg="#3498db", fg="white", **btn_style).pack(pady=10)

tk.Button(root, text="ЛАБОРАТОРНАЯ №2 (ФИЛЬТРАЦИЯ)", 
          command=lambda: run_lab("lab2_filters"), 
          bg="#27ae60", fg="white", **btn_style).pack(pady=10)

tk.Label(root, text="Меню вернется после закрытия окна лабы", 
         font=("Arial", 8, "italic"), fg="#bdc3c7", bg="#2c3e50").pack(side="bottom", pady=10)

root.mainloop()
