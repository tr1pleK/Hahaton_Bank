from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.config import settings

# Контекст для хеширования паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Проверяет пароль"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Хеширует пароль"""
    # Обрезаем пароль до 72 байт (ограничение bcrypt)
    # Простой подход: обрезаем строку до 72 символов
    # Это безопасно, так как большинство символов занимают 1 байт в UTF-8
    if len(password) > 72:
        password = password[:72]
    
    # Дополнительная проверка: если в байтах больше 72, обрезаем по байтам
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
        password = password_bytes.decode('utf-8', errors='ignore')
    
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Создает JWT токен (без срока действия - токен живет вечно)"""
    to_encode = data.copy()
    # Не добавляем поле exp, чтобы токен жил вечно
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    """Декодирует JWT токен (без проверки срока действия)"""
    try:
        # options={"verify_exp": False} отключает проверку срока действия
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM],
            options={"verify_exp": False}
        )
        return payload
    except JWTError:
        return None

