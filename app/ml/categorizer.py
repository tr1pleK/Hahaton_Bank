"""Категоризатор транзакций с использованием ML модели"""
from typing import Optional
from datetime import datetime
import pandas as pd
from app.models.category import TransactionCategory
from app.ml.transaction_classifier import TransactionClassifier

# Глобальный экземпляр классификатора (загружается один раз)
_classifier: Optional[TransactionClassifier] = None


def _get_classifier() -> TransactionClassifier:
    """Получить или создать экземпляр классификатора"""
    global _classifier
    if _classifier is None:
        _classifier = TransactionClassifier()
    return _classifier


def categorize_transaction(
    description: str, 
    amount: float, 
    is_expense: bool = True,
    date: Optional[datetime] = None
) -> TransactionCategory:
    """
    Определяет категорию транзакции с использованием ML модели
    
    Args:
        description: Описание транзакции (RefNo)
        amount: Сумма транзакции
        is_expense: Является ли транзакция расходом
        date: Дата транзакции
        
    Returns:
        Категория транзакции
    """
    classifier = _get_classifier()
    
    # Создаем DataFrame для предсказания (как в test_classifier.py)
    if date is None:
        date = datetime.now()
    
    withdrawal = amount if is_expense else 0.0
    deposit = amount if not is_expense else 0.0
    
    df = pd.DataFrame([{
        'Date': date,
        'RefNo': description or '',
        'Withdrawal': withdrawal,
        'Deposit': deposit,
        'Balance': 0.0  # Баланс неизвестен для одной транзакции
    }])
    
    category, probability = classifier.predict_from_dataframe(df)
    return category
