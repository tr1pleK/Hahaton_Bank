"""Простой категоризатор транзакций"""
from app.models.category import TransactionCategory


def categorize_transaction(description: str, amount: float) -> TransactionCategory:
    """Определяет категорию транзакции"""
    desc_lower = (description or "").lower()
    
    # Ключевые слова
    if any(word in desc_lower for word in ["продукты", "магазин", "еда"]):
        return TransactionCategory.PRODUCTS
    if any(word in desc_lower for word in ["транспорт", "метро", "такси"]):
        return TransactionCategory.TRANSPORT
    if any(word in desc_lower for word in ["кафе", "ресторан", "кофе"]):
        return TransactionCategory.CAFE
    if any(word in desc_lower for word in ["здоровье", "врач", "лекарство"]):
        return TransactionCategory.HEALTH
    
    # Большие суммы - возможно доход
    if amount > 10000:
        return TransactionCategory.SALARY
    
    return TransactionCategory.OTHER_EXPENSE
