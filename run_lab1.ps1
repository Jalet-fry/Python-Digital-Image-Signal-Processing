Write-Host "Запуск главного меню ЦОСиИ..." -ForegroundColor Cyan

# Путь к главному файлу
$MainScript = Join-Path $PSScriptRoot "main.py"

if (Test-Path $MainScript) {
    python $MainScript
} else {
    Write-Host "Ошибка: Файл main.py не найден!" -ForegroundColor Red
}
