$folders = @(
    "results/audio/lab3/source",
    "results/audio/lab3/processed",
    "results/graphs/lab3"
)

foreach ($folder in $folders) {
    if (!(Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder -Force
        Write-Host "Создана папка: $folder" -ForegroundColor Green
    }
}
