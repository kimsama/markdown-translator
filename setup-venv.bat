@echo off
echo Setting up virtual environment for AI Translator...
echo.

REM Create virtual environment
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate

REM Install requirements
pip install -r requirements.txt

echo.
echo Virtual environment 'venv' has been created and requirements installed.
echo.
echo To activate the virtual environment, run:
echo     venv\Scripts\activate
echo.
echo To use the translator, first activate the environment, then run:
echo     python translator.py -f path\to\file.md
echo     python translator.py -d path\to\directory
echo.
