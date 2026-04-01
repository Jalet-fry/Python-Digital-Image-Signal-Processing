# ==========================================================
# DSP Project Makefile - Unified Entry Point
# ==========================================================
PYTHON    = python
VARIANT   = 10

.PHONY: all lab1 lab2 lab3 clean install convert help

# По умолчанию запускаем общее меню
all:
	@$(PYTHON) main.py

help:
	@echo "Доступные команды:"
	@echo "  make         - Запуск общего меню выбора лаб"
	@echo "  make lab1    - Быстрый запуск Лабораторной №1"
	@echo "  make lab2    - Быстрый запуск Лабораторной №2"
	@echo "  make lab3    - Быстрый запуск Лабораторной №3"
	@echo "  make clean   - Очистка проекта (удаление core/signals и кэша)"
	@echo "  make install - Установка библиотек"

lab1:
	@$(PYTHON) labs/lab1_instruments.py --variant $(VARIANT)

lab2:
	@$(PYTHON) labs/lab2_filters.py --variant $(VARIANT)

lab3:
	@$(PYTHON) labs/lab3_speech.py --variant $(VARIANT)

convert:
	@$(PYTHON) labs/convert_audio.py

install:
	@$(PYTHON) -m pip install -r requirements.txt

clean:
	@echo "Очистка проекта от старой архитектуры и мусора..."
	@cmd /c cleanup_old_files.bat
	@$(PYTHON) -c "import shutil, os; shutil.rmtree('results/debug_logs', ignore_errors=True)"
	@echo "Готово."
