@echo off
echo ==========================================================
echo [CLEANUP] Start cleaning up the project...
echo ==========================================================

echo [1/3] Removing old core/signals directory (Architecture migration)...
if exist "C:\Univer\BSUIR_LABS\6_term\ЦОСиИ\PythonDSP\core\signals" (
    rmdir /s /q "C:\Univer\BSUIR_LABS\6_term\ЦОСиИ\PythonDSP\core\signals"
    echo [OK] core/signals removed.
)

echo [2/3] Removing redundant root scripts...
del "C:\Univer\BSUIR_LABS\6_term\ЦОСиИ\PythonDSP\clean.bat" 2>nul
del "C:\Univer\BSUIR_LABS\6_term\ЦОСиИ\PythonDSP\clean_project.ps1" 2>nul
del "C:\Univer\BSUIR_LABS\6_term\ЦОСиИ\PythonDSP\create_folders.ps1" 2>nul
del "C:\Univer\BSUIR_LABS\6_term\ЦОСиИ\PythonDSP\main.py" 2>nul
del "C:\Univer\BSUIR_LABS\6_term\ЦОСиИ\PythonDSP\Makefile.bak" 2>nul

echo [3/3] Clearing Python cache (__pycache__)...
for /d /r "C:\Univer\BSUIR_LABS\6_term\ЦОСиИ\PythonDSP" %%d in (__pycache__) do (
    if exist "%%d" (
        rmdir /s /q "%%d"
    )
)

echo ==========================================================
echo [DONE] Project is now clean and follow the Logic-UI-Core structure.
echo [INFO] Use 'make lab1', 'make lab2' or 'make lab3' to run the project.
echo ==========================================================
pause
