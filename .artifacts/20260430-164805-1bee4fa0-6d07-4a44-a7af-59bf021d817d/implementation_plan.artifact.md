# План реализации Лабораторной №4: kNN-VC и клонирование голоса

Цель: Привести проект в полное соответствие с заданием (kNN-VC, Zero-Shot TTS) и устранить "эффект робота". Реализация будет работать локально на CPU.

## Предлагаемые изменения

### [AI Logic]

#### [lab4_logic.py](file:///C:/Univer/BSUIR_LABS/6_term/ЦОСиИ/PythonDSP/lab4_final/project_files/lab4_logic.py)
- **Замена VC**: Вместо Whisper+Silero внедряем реальный алгоритм **kNN-VC**.
    - Использование модели **WavLM-Base-Plus** для извлечения признаков.
    - Реализация поиска k-ближайших соседей (k-Nearest Neighbors) в пространстве признаков целевого голоса.
    - Синтез через Mel-Inversion (Griffin-Lim или HiFi-GAN) для передачи тембра.
- **Исправление TTS**:
    - Починить загрузку дообученного вокодера (обработка несовпадения ключей `state_dict`).
    - Убедиться, что `SpeechT5` корректно использует эмбеддинги спикера для Zero-Shot синтеза.
- **Оптимизация**: Модели загружаются один раз и хранятся в памяти.

### [UI Controller]

#### [lab4_ai.py](file:///C:/Univer/BSUIR_LABS/6_term/ЦОСиИ/PythonDSP/lab4_final/project_files/lab4_ai.py)
- Исправить пути к `source_audio_lab3` (корень проекта).
- Передавать параметр `k` из слайдера в логику VC.
- Добавить индикацию процесса "Searching Neighbors..." в статус-бар.

### [Evaluation Scripts]

#### [evaluate_resources.py](file:///C:/Univer/BSUIR_LABS/6_term/ЦОСиИ/PythonDSP/lab4_final/evaluate_resources.py)
- Обновить скрипт, чтобы он тестировал актуальную kNN-VC логику вместо старой.

#### [test_minimal_length.py](file:///C:/Univer/BSUIR_LABS/6_term/ЦОСиИ/PythonDSP/lab4_final/test_minimal_length.py)
- Настроить замер качества в зависимости от количества извлеченных векторов WavLM.

---

## План верификации

### Автоматизированные тесты
- Запуск `standalone_lab4`.
- Проверка генерации `results/lab4_vc.wav`. Файл должен звучать голосом Target, а не робота.
- Проверка логов: отсутствие ошибок `UNEXPECTED key` при загрузке вокодера.

### Ручная проверка
1. Выбрать Source (свой голос) и Target (чужой голос).
2. Нажать `CONVERT VOICE`.
3. Убедиться, что результат имеет интонацию Source, но тембр Target.
4. Проверить Task 2.1 (Logs) — график должен строиться на основе `hifigan_train_log.csv`.
