Write-Host "Установка зависимостей проекта..." -ForegroundColor Green
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Write-Host "`n✅ Зависимости установлены!" -ForegroundColor Green
Read-Host "Нажмите Enter для выхода"

