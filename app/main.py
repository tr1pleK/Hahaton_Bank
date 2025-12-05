from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.exc import OperationalError
import time
from app.database import engine, Base
from app.config import settings


# Функция для ожидания готовности БД
def wait_for_db(max_retries=30, delay=2):
    """Ожидает готовности базы данных"""
    for i in range(max_retries):
        try:
            with engine.connect() as conn:
                conn.close()
            print("✅ База данных подключена!")
            return True
        except OperationalError as e:
            if i < max_retries - 1:
                print(f"⏳ Ожидание базы данных... ({i+1}/{max_retries})")
                time.sleep(delay)
            else:
                print(f"⚠️  Не удалось подключиться к базе данных: {e}")
                return False
    return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Startup
    print("🚀 Запуск приложения...")
    
    if not settings.SKIP_DB_CHECK:
        db_connected = wait_for_db()
        if db_connected:
            try:
                Base.metadata.create_all(bind=engine)
                print("✅ Таблицы БД созданы!")
            except Exception as e:
                print(f"⚠️  Ошибка при создании таблиц: {e}")
        else:
            print("⚠️  Приложение запущено без подключения к БД. Некоторые функции могут быть недоступны.")
    else:
        print("⏭️  Проверка БД пропущена (SKIP_DB_CHECK=True)")
    
    yield
    # Shutdown
    print("👋 Остановка приложения...")


# Создаем приложение FastAPI
app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "message": "Finance Analysis API",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья приложения"""
    return {"status": "healthy"}

