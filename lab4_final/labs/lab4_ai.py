#!/usr/bin/env python3
"""
Лабораторная работа №4 - Voice Conversion & TTS
Запуск всех заданий из одного скрипта
"""

import os
import sys
import subprocess
import argparse

def run_script(script_name, description):
    print(f"\n{'='*60}")
    print(f"📢 {description}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script_name], cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f"⚠️ Ошибка при выполнении {script_name}")
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description='Lab 4: Voice Conversion')
    parser.add_argument('--variant', type=int, default=10, help='Номер варианта')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', '1', '2', '2.2', '2.3', 'tts', 'vc', 'length', 'resources'],
                        help='Какой пункт запустить')
    args = parser.parse_args()

    print(f"🎤 Лабораторная работа №4 (вариант {args.variant})")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.task in ['all', '1', 'tts']:
        run_script(os.path.join(base_dir, 'task1_tts_coqui.py'),
                   "Задание 1: Озвучка текста (Coqui TTS)")

    if args.task in ['all', '2', 'vc']:
        run_script(os.path.join(base_dir, 'task2_vc_yourtts.py'),
                   "Задание 2: Конвертация голоса")

    if args.task in ['all', '2.3', 'length']:
        run_script(os.path.join(base_dir, 'test_minimal_length.py'),
                   "Задание 2.3: Эксперимент с минимальной длиной")

    if args.task in ['all', '2.2', 'resources']:
        run_script(os.path.join(base_dir, 'evaluate_resources.py'),
                   "Задание 2.2: Оценка ресурсов")

    print("\n" + "="*60)
    print("✅ Лабораторная работа №4 выполнена!")
    print("Результаты сохранены в папке output/")
    print("="*60)

if __name__ == "__main__":
    main()