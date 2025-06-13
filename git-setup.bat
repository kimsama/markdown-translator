@echo off
echo Setting up Git repository...
echo.

REM Navigate to the project directory
cd /d D:\dev\ai-translater

REM Stage all files for commit
git add .

REM Make initial commit
git commit -m "Initial commit: Korean Markdown Translator application"

REM Display status
git status

echo.
echo Git setup complete!
echo.
echo Run the following commands if you want to connect to a remote repository:
echo git remote add origin https://github.com/yourusername/ai-translater.git
echo git push -u origin main
