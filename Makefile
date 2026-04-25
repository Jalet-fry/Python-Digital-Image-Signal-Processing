# ==========================================================
# DSP Project Makefile - Unified Entry Point
# ==========================================================
PYTHON      = "C:\Python310\python.exe"
PYTHON_LAB4 = venv_lab4\Scripts\python.exe
VARIANT     = 10

.PHONY: all lab1 lab2 lab3 lab4 setup_lab4 clean install convert help

# По умолчанию запускаем общее меню
all:
	@$(PYTHON) main.py

help:
	@echo "Доступные команды:"
	@echo "  make            - Запуск общего меню выбора лаб"
	@echo "  make lab1       - Быстрый запуск Лабораторной №1"
	@echo "  make lab2       - Быстрый запуск Лабораторной №2"
	@echo "  make lab3       - Быстрый запуск Лабораторной №3"
	@echo "  make lab4       - Быстрый запуск Лабораторной №4 (использует изолированное окружение)"
	@echo "  make setup_lab4  - Создать окружение и установить зависимости для Лабы 4"
	@echo "  make clean      - Очистка проекта (удаление core/signals и кэша)"
	@echo "  make install    - Установка библиотек для лаб 1-3"

lab1:
	@$(PYTHON) labs/lab1_instruments.py --variant $(VARIANT)

lab2:
	@$(PYTHON) labs/lab2_filters.py --variant $(VARIANT)

lab3:
	@$(PYTHON) labs/lab3_speech.py --variant $(VARIANT)

lab4:
	@if not exist venv_lab4 (echo "Окружение venv_lab4 не найдено. Запустите 'make setup_lab4'" && exit /b 1)
	@$(PYTHON_LAB4) labs/lab4_ai.py --variant $(VARIANT)

setup_lab4:
	@echo "Создаем виртуальное окружение venv_lab4..."
	$(PYTHON) -m venv venv_lab4
	@echo "Обновляем базовые инструменты сборки..."
	@$(PYTHON_LAB4) -m pip install --upgrade pip setuptools wheel
	@echo "Устанавливаем библиотеки для Лабы 4 из requirements_lab4.txt..."
	@$(PYTHON_LAB4) -m pip install -r requirements_lab4.txt
	@echo "-------------------------------------------------------"
	@echo "ГОТОВО! Окружение для Лабы 4 настроено отдельно."
	@echo "Теперь вы можете запускать её через 'make lab4'"

convert:
	@$(PYTHON) labs/convert_audio.py

install:
	@$(PYTHON) -m pip install -r requirements.txt

clean:
	@echo "Очистка проекта от старой архитектуры и мусора..."
	@cmd /c cleanup_old_files.bat
	@$(PYTHON) -c "import shutil, os; shutil.rmtree('results/debug_logs', ignore_errors=True)"
	@echo "Готово."
