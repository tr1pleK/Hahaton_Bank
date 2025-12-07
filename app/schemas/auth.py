from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    """Схема токена"""
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    """Схема запроса на вход"""
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    """Схема запроса на регистрацию"""
    email: EmailStr
    password: str
    full_name: str


class UserResponse(BaseModel):
    """Схема ответа с данными пользователя"""
    id: int
    email: str
    full_name: str
    balance: float
    
    class Config:
        from_attributes = True


class BalanceResponse(BaseModel):
    """Схема ответа с балансом пользователя"""
    balance: float
    user_id: int
