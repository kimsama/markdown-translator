@echo off
echo Memory-Optimized Markdown Korean Translator
echo -----------------------------------------

REM Check if virtual environment exists and activate it
if exist "venv\Scripts\activate.bat" (
    echo Using virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if API key is set
if "%OPENAI_API_KEY%"=="" (
    echo NOTE: OPENAI_API_KEY environment variable is not set
    echo The application will try to load it from the .env file
    echo If not found there, you will need to provide it using the -k parameter
)

python translator_optimized.py %*

echo.
echo Done!
