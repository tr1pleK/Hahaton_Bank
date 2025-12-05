from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.exc import OperationalError
import time
from app.database import engine, Base, SessionLocal
from app.config import settings
from app.api import auth


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
                
                # Создаем демо-пользователя если его нет
                from app.models.user import User
                from app.models.billing import Billing, UserBilling
                from app.utils.security import get_password_hash
                from datetime import date
                
                db = SessionLocal()
                try:
                    demo_email = "demo@finance.app"
                    existing_user = db.query(User).filter(User.email == demo_email).first()
                    if not existing_user:
                        # Создаем демо-пользователя
                        demo_user = User(
                            email=demo_email,
                            password=get_password_hash("demo123"),
                            full_name="Демо Пользователь",
                            balance=10000.00
                        )
                        db.add(demo_user)
                        db.commit()
                        db.refresh(demo_user)
                        
                        # Создаем бесплатный тариф
                        free_billing = db.query(Billing).filter(Billing.type == "free").first()
                        if not free_billing:
                            free_billing = Billing(type="free", price=0.00)
                            db.add(free_billing)
                            db.commit()
                            db.refresh(free_billing)
                        
                        # Привязываем пользователя к тарифу
                        user_billing = UserBilling(
                            user_id=demo_user.id,
                            billing_id=free_billing.id,
                            start_date=date.today(),
                            end_date=None,
                            is_active=True
                        )
                        db.add(user_billing)
                        db.commit()
                        print("✅ Демо-пользователь создан автоматически!")
                        print(f"   Email: {demo_email}, Пароль: demo123")
                except Exception as e:
                    db.rollback()
                    print(f"⚠️  Не удалось создать демо-пользователя: {e}")
                finally:
                    db.close()
                    
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


# Подключаем роутеры
app.include_router(auth.router)

