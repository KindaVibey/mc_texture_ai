@echo off
echo ===============================================
echo  Minecraft Texture AI - GUI Launcher
echo ===============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo.
    echo Please install Python 3.13 from python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo Python found!
echo.

REM Check if dependencies are installed
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo Dependencies not installed yet.
    echo Running installer...
    echo.
    python INSTALL.py
    echo.
    echo Please run this file again to start the GUI.
    pause
    exit /b 0
)

echo Launching GUI...
echo.
python run_gui.py

if errorlevel 1 (
    echo.
    echo ===============================================
    echo  An error occurred!
    echo ===============================================
    echo.
    pause
)