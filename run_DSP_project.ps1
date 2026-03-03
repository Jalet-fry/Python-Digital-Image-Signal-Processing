Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "   Запуск проекта ЦОСиИ (Вариант 10)   " -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# 1. Проверка и установка зависимостей
Write-Host "Проверка зависимостей..." -ForegroundColor Yellow
$ReqFile = Join-Path $PSScriptRoot "requirements.txt"
if (Test-Path $ReqFile) {
    # pip install -r requirements.txt --quiet --no-cache-dir
    # Но лучше просто --quiet, чтобы не сканировать заново если все ок
    pip install -r $ReqFile --quiet --disable-pip-version-check
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Библиотеки проверены/обновлены." -ForegroundColor Green
    } else {
        Write-Host "[!] Внимание: Ошибка при проверке библиотек через pip." -ForegroundColor Magenta
    }
}

# 2. Запуск главного меню
$MainScript = Join-Path $PSScriptRoot "main.py"
if (Test-Path $MainScript) {
    Write-Host "Запуск главного меню..." -ForegroundColor Cyan
    python $MainScript
} else {
    Write-Host "ОШИБКА: Файл main.py не найден в корне папки!" -ForegroundColor Red
    pause
}
