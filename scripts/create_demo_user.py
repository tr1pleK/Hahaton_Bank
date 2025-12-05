"""
Скрипт для создания демонстрационного пользователя
Использование: py scripts/create_demo_user.py
"""
import sys
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import SessionLocal, engine, Base
from app.models.user import User
from app.models.billing import Billing, UserBilling
from app.utils.security import get_password_hash
from datetime import date

# Демо-пользователь
DEMO_EMAIL = "demo@finance.app"
DEMO_PASSWORD = "demo123"
DEMO_NAME = "Демо Пользователь"


def create_demo_user():
    """Создает демонстрационного пользователя"""
    # Создаем таблицы если их нет
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Проверяем, существует ли уже демо-пользователь
        existing_user = db.query(User).filter(User.email == DEMO_EMAIL).first()
        if existing_user:
            print(f"✅ Демо-пользователь уже существует!")
            print(f"   Email: {DEMO_EMAIL}")
            print(f"   Пароль: {DEMO_PASSWORD}")
            return
        
        # Создаем демо-пользователя
        demo_user = User(
            email=DEMO_EMAIL,
            password=get_password_hash(DEMO_PASSWORD),
            full_name=DEMO_NAME,
            balance=10000.00
        )
        db.add(demo_user)
        db.commit()
        db.refresh(demo_user)
        
        # Создаем бесплатный тарифный план если его нет
        free_billing = db.query(Billing).filter(Billing.type == "free").first()
        if not free_billing:
            free_billing = Billing(
                type="free",
                price=0.00
            )
            db.add(free_billing)
            db.commit()
            db.refresh(free_billing)
        
        # Привязываем пользователя к бесплатному тарифу
        user_billing = UserBilling(
            user_id=demo_user.id,
            billing_id=free_billing.id,
            start_date=date.today(),
            end_date=None,  # Бессрочная подписка
            is_active=True
        )
        db.add(user_billing)
        db.commit()
        
        print("✅ Демонстрационный пользователь создан!")
        print(f"   Email: {DEMO_EMAIL}")
        print(f"   Пароль: {DEMO_PASSWORD}")
        print(f"   Имя: {DEMO_NAME}")
        print(f"   Баланс: {demo_user.balance}")
        print(f"   Тариф: {free_billing.type}")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Ошибка при создании демо-пользователя: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    create_demo_user()

