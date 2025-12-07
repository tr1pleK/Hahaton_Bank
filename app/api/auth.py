from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User
from app.models.billing import Billing, UserBilling
from app.schemas.auth import Token, LoginRequest, RegisterRequest, UserResponse, BalanceResponse
from app.utils.security import verify_password, create_access_token, get_password_hash
from app.dependencies import get_current_user
from datetime import date

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=Token)
async def login(login_data: LoginRequest, db: Session = Depends(get_db)):
    """Авторизация пользователя"""
    # Ищем пользователя по email
    user = db.query(User).filter(User.email == login_data.email).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль"
        )
    
    # Проверяем пароль
    if not verify_password(login_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль"
        )
    
    # Создаем токен
    access_token = create_access_token(data={"sub": str(user.id), "email": user.email})
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(register_data: RegisterRequest, db: Session = Depends(get_db)):
    """Регистрация нового пользователя"""
    # Проверяем, не существует ли уже пользователь с таким email
    existing_user = db.query(User).filter(User.email == register_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким email уже существует"
        )
    
    # Создаем нового пользователя
    hashed_password = get_password_hash(register_data.password)
    new_user = User(
        email=register_data.email,
        password=hashed_password,
        full_name=register_data.full_name,
        balance=0.00
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    # Привязываем к бесплатному тарифу
    free_billing = db.query(Billing).filter(Billing.type == "free").first()
    if not free_billing:
        free_billing = Billing(type="free", price=0.00)
        db.add(free_billing)
        db.commit()
        db.refresh(free_billing)
    
    user_billing = UserBilling(
        user_id=new_user.id,
        billing_id=free_billing.id,
        start_date=date.today(),
        end_date=None,
        is_active=True
    )
    db.add(user_billing)
    db.commit()
    
    return new_user


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Получить данные текущего пользователя"""
    return current_user


@router.get("/balance", response_model=BalanceResponse)
async def get_balance(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Получить баланс текущего пользователя"""
    # Обновляем данные пользователя из БД для получения актуального баланса
    db.refresh(current_user)
    return BalanceResponse(
        balance=float(current_user.balance),
        user_id=current_user.id
    )

