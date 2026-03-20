# Скрипт очистки временных файлов и логов проекта PythonDSP

$ProjectRoot = Get-Location

# 1. Удаление логов отладки (текстовые дампы массивов)
$DebugLogs = Join-Path $ProjectRoot "results\debug_logs"
if (Test-Path $DebugLogs) {
    Write-Host "Очистка логов отладки: $DebugLogs" -ForegroundColor Cyan
    Remove-Item -Path "$DebugLogs\*" -Recurse -Force
}

# 2. Удаление сгенерированных графиков (PNG)
$GraphsDir = Join-Path $ProjectRoot "results\graphs"
if (Test-Path $GraphsDir) {
    Write-Host "Очистка папки графиков: $GraphsDir" -ForegroundColor Cyan
    Remove-Item -Path "$GraphsDir\*" -Recurse -Force
}

# 3. Удаление сгенерированного аудио (WAV)
$AudioDir = Join-Path $ProjectRoot "results\audio"
if (Test-Path $AudioDir) {
    Write-Host "Очистка папки аудио: $AudioDir" -ForegroundColor Cyan
    Remove-Item -Path "$AudioDir\*" -Recurse -Force
}

# 4. Удаление кэша Python (__pycache__)
Write-Host "Очистка кэша Python (__pycache__)..." -ForegroundColor Cyan
Get-ChildItem -Path $ProjectRoot -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force

# 5. Удаление отчетов (txt)
$ReportsDir = Join-Path $ProjectRoot "results"
if (Test-Path $ReportsDir) {
    Write-Host "Очистка отчетов в results/*.txt" -ForegroundColor Cyan
    Get-ChildItem -Path $ReportsDir -Filter "*.txt" | Remove-Item -Force
}

Write-Host "Проект очищен!" -ForegroundColor Green
