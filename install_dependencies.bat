@echo off
echo Установка зависимостей проекта...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo ✅ Зависимости установлены!
pause

