"""Схемы для валидации транзакций"""
from pydantic import BaseModel, Field
from datetime import date
from typing import Optional


class TransactionCreate(BaseModel):
    """Схема для создания транзакции"""
    amount: float = Field(..., gt=0)
    description: Optional[str] = None
    date: date
    category: Optional[str] = None


class TransactionUpdate(BaseModel):
    """Схема для обновления транзакции"""
    amount: Optional[float] = Field(None, gt=0)
    description: Optional[str] = None
    date: Optional[date] = None
    category: Optional[str] = None
