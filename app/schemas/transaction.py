"""Схемы для валидации транзакций"""
from pydantic import BaseModel, Field
from datetime import date
from typing import Optional, List


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


class CSVTransactionInput(BaseModel):
    """Схема для входных данных транзакции в формате CSV"""
    Date: str
    RefNo: Optional[str] = None
    Withdrawal: Optional[float] = None
    Deposit: Optional[float] = None
    Balance: Optional[float] = None


class CSVTransactionPredictionResponse(BaseModel):
    """Схема для ответа с предсказанной категорией"""
    Date: str
    RefNo: Optional[str] = None
    Withdrawal: Optional[float] = None
    Deposit: Optional[float] = None
    Balance: Optional[float] = None
    Category: str
    Probability: float = Field(..., ge=0.0, le=1.0)
