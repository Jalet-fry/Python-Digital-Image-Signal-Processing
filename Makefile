# Настройки
PYTHON    = python
VARIANT   = 10
REQ_FILE  = requirements.txt
DEP_STAMP = .deps_checked

# Цвета (ANSI)
CYAN   = \033[0;36m
YELLOW = \033[1;33m
GREEN  = \033[0;32m
RED    = \033[0;31m
NC     = \033[0m

.PHONY: all lab1 lab2 lab3 convert clean install setup header

all: setup header
	@$(PYTHON) main.py

# Запуск Лаб
lab1: header
	@$(PYTHON) labs/lab1_instruments.py --variant $(VARIANT)

lab2: header
	@$(PYTHON) labs/lab2_filters.py --variant $(VARIANT)

lab3: header
	@$(PYTHON) labs/lab3_speech.py --variant $(VARIANT)

# Команда для конвертации аудио (Лаба 3)
convert: header
	@$(PYTHON) -c "print('$(YELLOW)>>> Запуск конвертера аудио (WAV 16kHz Mono)...$(NC)')"
	@$(PYTHON) labs/convert_audio.py
	@$(PYTHON) -c "print('$(GREEN)>>> [OK] Конвертация завершена.$(NC)')"

# Установка зависимостей
install:
	@$(PYTHON) -c "print('$(YELLOW)>>> Установка библиотек (фиксация версий для DeepFilterNet)...$(NC)')"
	$(PYTHON) -m pip install -r $(REQ_FILE) --force-reinstall
	@$(PYTHON) -c "from pathlib import Path; Path('$(DEP_STAMP)').touch()"
	@$(PYTHON) -c "print('$(GREEN)[OK] Все библиотеки установлены.$(NC)')"

setup:
	@$(PYTHON) -c "import importlib.util; libs=['torch', 'torchaudio', 'numpy', 'matplotlib', 'scipy', 'librosa', 'sounddevice', 'pydub', 'df']; \
	missing=[l for l in libs if importlib.util.find_spec(l) is None]; \
	print('$(GREEN)>>> [OK] Все зависимости найдены.$(NC)') if not missing else print('$(RED)>>> [!] Отсутствуют библиотеки: ' + ', '.join(missing) + '. \n>>> Запустите: make install$(NC)')"

header:
	@$(PYTHON) -c "print('$(CYAN)>>> DSP STATION (Вариант $(VARIANT)) | Project Manager$(NC)')"

clean:
	@$(PYTHON) -c "print('$(YELLOW)Очистка временных файлов и логов...$(NC)')"
	@$(PYTHON) -c "import shutil, os; [shutil.rmtree(os.path.join('results', d), ignore_errors=True) for d in ['debug_logs', 'logs/lab3', 'audio/lab3/processed', 'audio/lab3/noisy']]"
	@$(PYTHON) -c "print('$(GREEN)Готово.$(NC)')"
