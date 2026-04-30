import os
import sys

# Попытка импортировать библиотеку для загрузки
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Ошибка: Не установлена библиотека huggingface_hub. Установите её: pip install huggingface_hub")
    sys.exit(1)

# Фикс для быстрой загрузки через зеркало в РБ/РФ
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

models = [
    "microsoft/wavlm-base-plus",
    "microsoft/speecht5_tts",
    "microsoft/speecht5_hifigan"
]

# Путь к кэшу внутри проекта
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cache_dir = os.path.join(base_dir, "core", "models", "hf_cache")
os.makedirs(cache_dir, exist_ok=True)

print(f"===========================================================")
print(f"ЗАГРУЗЧИК НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ (Лабораторная №4)")
print(f"Место сохранения: {cache_dir}")
print(f"Общий объем: ~1.1 ГБ")
print(f"===========================================================")

try:
    for model in models:
        print(f"\n[+] Начало загрузки: {model}")
        snapshot_download(
            repo_id=model,
            cache_dir=cache_dir,
            local_files_only=False,
            resume_download=True  # Позволяет докачивать при обрыве
        )
    print(f"\n[!!!] УСПЕХ: Все модели загружены.")
    print(f"Теперь вы можете запускать 'make lab4', всё будет работать офлайн.")
except Exception as e:
    print(f"\n[!] Ошибка при загрузке: {e}")
    print("Попробуйте запустить скрипт еще раз. Он поддерживает докачку.")
