# Finance Analysis API - Backend

Backend для системы интеллектуального анализа личных финансов.

## Технологии

- Python 3.11
- FastAPI
- SQLAlchemy
- PostgreSQL
- Docker & Docker Compose

## Быстрый старт

### С Docker Compose (рекомендуется)

1. Скопируйте `.env.example` в `.env` и при необходимости измените настройки:
```bash
cp .env.example .env
```

2. Запустите контейнеры:
```bash
docker-compose up -d
```

3. Приложение будет доступно по адресу: http://localhost:8000
4. Документация API: http://localhost:8000/docs

### Без Docker

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Настройте PostgreSQL и создайте базу данных

3. Создайте файл `.env` с настройками:
```
DATABASE_URL=postgresql://user:password@localhost:5432/finance_db
SECRET_KEY=your-secret-key
```

4. Запустите приложение:
```bash
uvicorn app.main:app --reload
```

## Структура проекта

```
backend/
├── app/
│   ├── api/          # API endpoints
│   ├── models/       # SQLAlchemy модели
│   ├── schemas/      # Pydantic схемы
│   ├── services/     # Бизнес-логика
│   ├── ml/           # ML модели
│   ├── config.py     # Конфигурация
│   ├── database.py   # Настройка БД
│   └── main.py       # Точка входа
├── ml_models/        # Сохраненные ML модели
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## API Endpoints

- `GET /` - Информация о API
- `GET /health` - Проверка здоровья приложения
- `GET /docs` - Интерактивная документация (Swagger)

## Команды Docker

```bash
# Запуск
docker-compose up -d

# Остановка
docker-compose down

# Просмотр логов
docker-compose logs -f backend

# Пересборка
docker-compose build --no-cache

# Выполнение команд в контейнере
docker-compose exec backend bash
```


