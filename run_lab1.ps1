# Переходим в директорию скрипта (корень проекта)
Set-Location -Path $PSScriptRoot

# Запускаем Python для первой лабораторной
Write-Host "Запуск Лабораторной №1..." -ForegroundColor Cyan
python ./labs/lab1_instruments.py
