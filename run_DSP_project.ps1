Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "   Запуск проекта ЦОСиИ (Вариант 10)   " -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Путь к главному меню
$MainScript = Join-Path $PSScriptRoot "main.py"

if (Test-Path $MainScript) {
    python $MainScript
} else {
    Write-Host "ОШИБКА: Файл main.py не найден в корне папки!" -ForegroundColor Red
    pause
}
