# Настройки
PYTHON    = python
VARIANT   = 10
REQ_FILE  = requirements.txt
DEP_STAMP = .deps_checked

# Цвета (ANSI)
CYAN   = \033[0;36m
YELLOW = \033[1;33m
GREEN  = \033[0;32m
NC     = \033[0m

.PHONY: all lab1 lab2 lab3 clean install header setup

all: header
	@$(PYTHON) main.py

lab1: header
	@$(PYTHON) labs/lab1_instruments.py --variant $(VARIANT)

lab2: header
	@$(PYTHON) labs/lab2_filters.py --variant $(VARIANT)

lab3: header
	@$(PYTHON) labs/lab3_speech.py --variant $(VARIANT)

install:
	@$(PYTHON) -c "print('$(YELLOW)>>> Установка библиотек...$(NC)')"
	@$(PYTHON) -m pip install -r $(REQ_FILE) --disable-pip-version-check
	@$(PYTHON) -c "from pathlib import Path; Path('$(DEP_STAMP)').touch()"
	@$(PYTHON) -c "print('$(GREEN)[OK] Готово.$(NC)')"

setup:
	@$(PYTHON) -c "import sys; libs=['numpy', 'matplotlib', 'scipy', 'librosa', 'sounddevice', 'numba', 'pesq', 'pystoi']; \
	missing=[l for l in libs if __import__('importlib').util.find_spec(l) is None]; \
	print('$(GREEN)>>> [OK] Библиотеки на месте.$(NC)') if not missing else print('$(YELLOW)>>> [!] Отсутствуют: ' + ', '.join(missing) + '. Запустите \"make install\"$(NC)')"

header:
	@$(PYTHON) -c "print('$(CYAN)>>> DSP STATION (Вариант $(VARIANT)) [NJIT: ON]$(NC)')"

clean:
	@$(PYTHON) -c "print('$(YELLOW)Очистка результатов...$(NC)')"
	@$(PYTHON) -c "import shutil, os; [shutil.rmtree(os.path.join('results', d), ignore_errors=True) for d in ['debug_logs', 'audio/lab3/processed', 'audio/lab3/noisy']]"
	@$(PYTHON) -c "print('$(GREEN)Готово.$(NC)')"
