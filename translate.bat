@echo off
echo Markdown Korean Translator
echo -----------------------

REM Check if API key is set
if "%ANTHROPIC_API_KEY%"=="" (
    echo NOTE: ANTHROPIC_API_KEY environment variable is not set
    echo The application will try to load it from the .env file
    echo If not found there, you will need to provide it using the -k parameter
)

python translator.py %*

echo.
echo Done!
