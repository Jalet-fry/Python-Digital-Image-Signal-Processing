Write-Host "=======================================" -ForegroundColor Cyan
Write-Host "   Запуск проекта ЦОСиИ (Вариант 10)   " -ForegroundColor Cyan
Write-Host "   [Ускорение: JIT Numba Enabled]      " -ForegroundColor Yellow
Write-Host "=======================================" -ForegroundColor Cyan

# 1. Проверка версии Python
$PyVer = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ОШИБКА] Python не найден в системе!" -ForegroundColor Red
    pause
    exit
}
Write-Host "[INFO] Используется $PyVer" -ForegroundColor Gray

# 2. Умная проверка и установка зависимостей
Write-Host "Проверка библиотек (это может занять время)..." -ForegroundColor Yellow
$ReqFile = Join-Path $PSScriptRoot "requirements.txt"

if (Test-Path $ReqFile) {
    # Используем --no-warn-script-location чтобы не спамить в консоль
    python -m pip install -r $ReqFile --quiet --disable-pip-version-check --no-warn-script-location

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] Все зависимости (NumPy, SciPy, Numba) готовы к работе." -ForegroundColor Green
    } else {
        Write-Host "[!] Внимание: Ошибка при установке библиотек. Проверьте интернет." -ForegroundColor Magenta
    }
}

# 3. Запуск главного меню
$MainScript = Join-Path $PSScriptRoot "main.py"
if (Test-Path $MainScript) {
    Write-Host "---------------------------------------" -ForegroundColor DarkGray
    Write-Host "Запуск GUI приложения..." -ForegroundColor Cyan
    # Запускаем без вывода лишних логов в консоль
    python $MainScript
} else {
    Write-Host "ОШИБКА: Файл main.py не найден!" -ForegroundColor Red
    pause
}