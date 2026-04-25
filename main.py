import sys
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from core.utils.themes import UIColors

# Импортируем функции main из лаб (с переименованием)
from labs.lab1_instruments import main as run_lab1
# Лаб 2 и 3 импортируем только при вызове, чтобы не грузить torch сразу

def show_menu():
    UIColors.apply_style()
    fig = plt.figure(num='BSUIR DSP STATION', figsize=(10, 7))
    fig.patch.set_facecolor(UIColors.BG_DARK)
    
    plt.text(0.5, 0.9, "ЦОСиИ: ЦИФРОВАЯ СТАНЦИЯ", fontsize=18, ha='center', color=UIColors.TEXT_ACCENT, weight='bold')
    plt.text(0.5, 0.84, "Выберите лабораторную работу", fontsize=10, ha='center', color=UIColors.TEXT_MAIN)
    
    # Лаб 1
    ax1 = plt.axes([0.2, 0.65, 0.6, 0.1])
    btn1 = Button(ax1, 'ЛАБОРАТОРНАЯ №1: СИНТЕЗ', color=UIColors.BTN_PLAY, hovercolor=UIColors.TEXT_ACCENT)
    btn1.on_clicked(lambda x: run_lab1())
    
    # Лаб 2
    ax2 = plt.axes([0.2, 0.52, 0.6, 0.1])
    btn2 = Button(ax2, 'ЛАБОРАТОРНАЯ №2: ФИЛЬТРАЦИЯ', color=UIColors.BTN_RUN, hovercolor=UIColors.TEXT_ACCENT)
    def call_lab2(event):
        from labs.lab2_filters import main as run_lab2
        run_lab2()
    btn2.on_clicked(call_lab2)
    
    # Лаб 3
    ax3 = plt.axes([0.2, 0.39, 0.6, 0.1])
    btn3 = Button(ax3, 'ЛАБОРАТОРНАЯ №3: РЕЧЬ', color='#4C51BF', hovercolor=UIColors.TEXT_ACCENT)
    def call_lab3(event):
        from labs.lab3_speech import main as run_lab3
        run_lab3()
    btn3.on_clicked(call_lab3)

    # Лаб 4 (AI)
    ax4 = plt.axes([0.2, 0.26, 0.6, 0.1])
    btn4 = Button(ax4, 'ЛАБОРАТОРНАЯ №4: AI AUDIO', color='#805AD5', hovercolor=UIColors.TEXT_ACCENT)
    def call_lab4(event):
        import subprocess
        python_lab4 = os.path.join(os.getcwd(), "venv_lab4", "Scripts", "python.exe")
        if os.path.exists(python_lab4):
            print("Запуск Лабораторной №4 через venv_lab4...")
            subprocess.run([python_lab4, "labs/lab4_ai.py", "--task", "0"])
        else:
            print("ОШИБКА: Виртуальное окружение venv_lab4 не найдено.")
            print("Пожалуйста, запустите 'make setup_lab4' в терминале.")
    btn4.on_clicked(call_lab4)

    # Выход
    ax_exit = plt.axes([0.4, 0.15, 0.2, 0.06])
    btn_exit = Button(ax_exit, 'ВЫХОД', color='#C53030', hovercolor='#9B2C2C')
    btn_exit.on_clicked(lambda x: plt.close('all'))
    btn_exit.label.set_color('white')

    fig.buttons = [btn1, btn2, btn3, btn_exit]
    plt.show()

if __name__ == "__main__":
    show_menu()
