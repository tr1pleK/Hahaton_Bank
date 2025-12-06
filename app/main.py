from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from sqlalchemy.exc import OperationalError
import time
import sys
import traceback

# Увеличиваем лимит рекурсии для более детального traceback
sys.setrecursionlimit(5000)

from app.database import engine, Base, SessionLocal
from app.config import settings

# Логирование импортов для отладки
print("📦 Импорт модулей...")
print(f"📦 Лимит рекурсии установлен: {sys.getrecursionlimit()}")

try:
    print("  → Импорт auth...")
    from app.api import auth
    print("  ✅ auth импортирован")
except RecursionError as e:
    print(f"  ❌ RecursionError при импорте auth!")
    exc_lines = traceback.format_exc().split('\n')
    print(f"  Глубина рекурсии: {len(exc_lines)}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"  ❌ Ошибка импорта auth: {e}")
    print(f"  Тип ошибки: {type(e).__name__}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("  → Импорт transactions...")
    print("    → Начало импорта модуля transactions...")
    # Пробуем импортировать по частям для диагностики
    try:
        import app.api.transactions as transactions_module
        print("    → Модуль импортирован, проверяем содержимое...")
        print(f"    → Файл модуля: {getattr(transactions_module, '__file__', 'неизвестно')}")
        
        # Пробуем выполнить код модуля вручную
        import importlib
        importlib.reload(transactions_module)
        
        if hasattr(transactions_module, 'router'):
            print(f"  ✅ router найден: {type(transactions_module.router)}")
            transactions = transactions_module
        else:
            print(f"  ❌ router НЕ найден после reload!")
            print(f"  Доступные атрибуты: {[a for a in dir(transactions_module) if not a.startswith('_')]}")
            # Пробуем создать роутер вручную
            print("  → Пробуем создать роутер вручную...")
            from fastapi import APIRouter
            transactions_module.router = APIRouter(prefix="/transactions", tags=["transactions"])
            transactions = transactions_module
            print("  ✅ Роутер создан вручную")
    except Exception as import_error:
        print(f"  ❌ Ошибка при импорте/проверке модуля: {import_error}")
        print(f"  Тип ошибки: {type(import_error).__name__}")
        traceback.print_exc()
        raise
    
    print("  ✅ transactions импортирован")
except RecursionError as e:
    print(f"  ❌ RecursionError при импорте transactions!")
    exc_lines = traceback.format_exc().split('\n')
    print(f"  Глубина рекурсии: {len(exc_lines)}")
    print("  Полный traceback:")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"  ❌ Ошибка импорта transactions: {e}")
    print(f"  Тип ошибки: {type(e).__name__}")
    print("  Полный traceback:")
    traceback.print_exc()
    sys.exit(1)


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
    description="API для анализа финансовых транзакций с ML классификацией",
    version="1.0.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Настройка OpenAPI схемы для Bearer token
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    from fastapi.openapi.utils import get_openapi
    
    openapi_schema = get_openapi(
        title=settings.APP_NAME,
        version="1.0.0",
        description="API для анализа финансовых транзакций с ML классификацией",
        routes=app.routes,
    )
    
    # HTTPBearer автоматически создаст Bearer схему, но мы можем улучшить описание
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "securitySchemes" not in openapi_schema["components"]:
        openapi_schema["components"]["securitySchemes"] = {}
    
    # Убеждаемся, что Bearer схема настроена правильно
    openapi_schema["components"]["securitySchemes"]["Bearer"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Введите Bearer token, полученный при авторизации через /auth/login. Просто вставьте токен без слова 'Bearer'"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Переопределяем openapi для улучшения Bearer token схемы
app.openapi = custom_openapi

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


# Глобальный обработчик ошибок валидации
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Обработчик ошибок валидации данных
    Возвращает понятные сообщения об ошибках для фронтенда
    """
    errors = []
    for error in exc.errors():
        # Формируем путь к полю
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        
        # Получаем сообщение об ошибке
        message = error.get("msg", "Ошибка валидации")
        
        # Улучшаем сообщения для числовых полей
        if "value is not a valid" in message.lower() and "float" in message.lower():
            message = f"Поле '{field_path}' должно быть числом. Получено некорректное значение."
        elif "value is not a valid" in message.lower():
            message = f"Поле '{field_path}' имеет некорректный формат."
        elif "не может быть отрицательным" in message.lower() or "greater than or equal to" in message.lower():
            # Сообщение уже хорошее, оставляем как есть или улучшаем
            if "greater than or equal to" in message.lower():
                message = f"Поле '{field_path}' не может быть отрицательным. Укажите положительное число или ноль."
        
        errors.append({
            "field": field_path,
            "message": message,
            "type": error.get("type", "validation_error"),
            "input": error.get("input")
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": errors,
            "error": "Ошибка валидации данных",
            "message": "Проверьте правильность введенных данных. Все числовые поля (Withdrawal, Deposit, Balance) должны содержать числа."
        }
    )


# Подключаем роутеры
print("🔌 Подключение роутеров...")
try:
    print("  → Подключение auth.router...")
    app.include_router(auth.router)
    print("  ✅ auth.router подключен")
except Exception as e:
    print(f"  ❌ Ошибка подключения auth.router: {e}")
    traceback.print_exc()

try:
    print("  → Подключение transactions.router...")
    app.include_router(transactions.router)
    print("  ✅ transactions.router подключен")
except Exception as e:
    print(f"  ❌ Ошибка подключения transactions.router: {e}")
    print(f"  Тип ошибки: {type(e).__name__}")
    traceback.print_exc()
    raise

