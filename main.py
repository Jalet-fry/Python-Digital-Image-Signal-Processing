import sys
import os
import subprocess
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from core.utils.themes import UIColors

# Словарь для отслеживания активных процессов
active_processes = {}

def run_lab(name, script):
    # Проверяем, не запущен ли уже этот процесс
    if name in active_processes:
        if active_processes[name].poll() is None:
            print(f">>> [UI] Лабораторная {name} уже запущена.")
            return
    
    print(f">>> [UI] Запуск {name}...")
    active_processes[name] = subprocess.Popen([sys.executable, script])

def show_menu():
    UIColors.apply_style()
    fig = plt.figure(figsize=(8, 6))
    fig.canvas.manager.set_window_title('BSUIR DSP Manager')
    
    plt.text(0.5, 0.85, "DSP STATION", fontsize=20, ha='center', color=UIColors.TEXT_ACCENT, weight='bold')
    plt.text(0.5, 0.78, "Выберите лабораторную работу ",
             fontsize=10, ha='center', color=UIColors.TEXT_DIM)
    
    # Лаб 1
    ax1 = plt.axes([0.2, 0.6, 0.6, 0.08])
    btn1 = Button(ax1, 'ЛАБ №1: Инструменты', color=UIColors.LAB1['x'], hovercolor=UIColors.TEXT_ACCENT)
    btn1.on_clicked(lambda x: run_lab("Lab1", "labs/lab1_instruments.py"))
    
    # Лаб 2
    ax2 = plt.axes([0.2, 0.48, 0.6, 0.08])
    btn2 = Button(ax2, 'ЛАБ №2: Фильтрация', color=UIColors.LAB2['ma'], hovercolor=UIColors.TEXT_ACCENT)
    btn2.on_clicked(lambda x: run_lab("Lab2", "labs/lab2_filters.py"))
    
    # Лаб 3
    ax3 = plt.axes([0.2, 0.36, 0.6, 0.08])
    btn3 = Button(ax3, 'ЛАБ №3: Речь', color=UIColors.LAB3['metrics'], hovercolor=UIColors.TEXT_ACCENT)
    btn3.on_clicked(lambda x: run_lab("Lab3", "labs/lab3_speech.py"))

    fig.buttons = [btn1, btn2, btn3]
    plt.show()

if __name__ == "__main__":
    show_menu()
