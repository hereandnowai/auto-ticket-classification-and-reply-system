@echo off
REM ============================================================
REM Shanyan AI - Complete System Setup & Startup Script (Windows)
REM ============================================================
REM This script will:
REM   1. Create Python virtual environment (if not exists)
REM   2. Install Python dependencies
REM   3. Install Node.js dependencies
REM   4. Train the model (if not already trained)
REM   5. Start the backend server
REM   6. Start the frontend server
REM   7. Open the browser
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ================================================================
echo           Shanyan AI - Auto Setup and Start
echo ================================================================
echo.

REM Get script directory
set "SCRIPT_DIR=%~dp0"
set "BACKEND_DIR=%SCRIPT_DIR%backend"
set "FRONTEND_DIR=%SCRIPT_DIR%frontend"
set "VENV_DIR=%BACKEND_DIR%\.venv"

REM ============================================================
REM Step 1: Create Python Virtual Environment
REM ============================================================
echo [Step 1] Setting up Python Virtual Environment...

if exist "%VENV_DIR%" goto :venv_exists
echo    Creating virtual environment...
python -m venv "%VENV_DIR%"
echo    Virtual environment created
goto :venv_done

:venv_exists
echo    Virtual environment already exists

:venv_done
REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"
echo    Virtual environment activated
echo.

REM ============================================================
REM Step 2: Install Python Dependencies
REM ============================================================
echo [Step 2] Installing Python Dependencies...

set "DEPS_MARKER=%VENV_DIR%\.deps_installed"

if exist "%DEPS_MARKER%" goto :deps_exists
echo    Installing dependencies - first time setup...
python -m pip install --quiet --upgrade pip --no-warn-script-location
python -m pip install --quiet -r "%BACKEND_DIR%\requirements.txt" --no-warn-script-location
echo. > "%DEPS_MARKER%"
echo    Python dependencies installed
goto :deps_done

:deps_exists
echo    Python dependencies already installed

:deps_done
echo.

REM ============================================================
REM Step 3: Check for .env file
REM ============================================================
echo [Step 3] Checking Environment Configuration...

if exist "%BACKEND_DIR%\.env" goto :env_exists
echo    .env file not found!
echo    Creating .env from .env.example...
copy "%BACKEND_DIR%\.env.example" "%BACKEND_DIR%\.env" > nul
echo    Please edit backend\.env and add your GEMINI_API_KEY
echo    Get your API key from: https://aistudio.google.com/api-keys
goto :env_done

:env_exists
echo    .env file found

:env_done
echo.

REM ============================================================
REM Step 4: Train Model (if not already trained)
REM ============================================================
echo [Step 4] Checking ML Model...

set "MODEL_PATH=%BACKEND_DIR%\training\models\fine_tuned_bert"

if exist "%MODEL_PATH%\config.json" goto :model_exists
echo    Training DistilBERT model - this may take 10-30 minutes...
cd "%SCRIPT_DIR%"
python "%BACKEND_DIR%\training\train.py"
echo    Model training complete
goto :model_done

:model_exists
echo    Trained model found

:model_done
echo.

REM ============================================================
REM Step 5: Install Frontend Dependencies
REM ============================================================
echo [Step 5] Installing Frontend Dependencies...

cd "%FRONTEND_DIR%"
if exist "node_modules" goto :npm_exists
echo    Running npm install...
npm install --silent
echo    Frontend dependencies installed
goto :npm_done

:npm_exists
echo    Node modules already installed

:npm_done
echo.

REM ============================================================
REM Step 6: Start Backend Server
REM ============================================================
echo [Step 6] Starting Backend Server...

cd "%BACKEND_DIR%"
start "Shanyan Backend" cmd /k "call "%VENV_DIR%\Scripts\activate.bat" && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
echo    Backend starting in new window...
timeout /t 5 /nobreak > nul
echo.

REM ============================================================
REM Step 7: Initialize Demo Users
REM ============================================================
echo [Step 7] Initializing Demo Users...
curl -s -X POST http://localhost:8000/api/init-users > nul 2>&1
echo    Demo users ready
echo.

REM ============================================================
REM Step 8: Start Frontend Server
REM ============================================================
echo [Step 8] Starting Frontend Server...

cd "%FRONTEND_DIR%"
start "Shanyan Frontend" cmd /k "npm run dev"
echo    Frontend starting in new window...
timeout /t 3 /nobreak > nul
echo.

REM ============================================================
REM Step 9: Open Browser
REM ============================================================
echo [Step 9] Opening Browser...
timeout /t 2 /nobreak > nul
start http://localhost:5173
echo    Browser opened
echo.

REM ============================================================
REM Success Message
REM ============================================================
echo ================================================================
echo              System Ready and Running!
echo ================================================================
echo.
echo Demo Accounts:
echo    Admin:        admin / admin123
echo    Client:       client1 / client123 (Rajesh Kumar)
echo    Tech Support: tech1 / tech123 (Priya Sharma)
echo    Accounting:   acc1 / acc123 (Amit Patel)
echo    Sales:        sales1 / sales123 (Sneha Reddy)
echo.
echo URLs:
echo    Frontend:  http://localhost:5173
echo    Backend:   http://localhost:8000
echo    API Docs:  http://localhost:8000/docs
echo.
echo Created by Shankar Narayanan, Student, Dr MGR University
echo.
pause