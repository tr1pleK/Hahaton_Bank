from enum import Enum


class TransactionCategory(str, Enum):
    """Категории транзакций"""
    # Расходы
    PRODUCTS = "Продукты"
    TRANSPORT = "Транспорт"
    CAFE = "Кафе и рестораны"
    HEALTH = "Здоровье"
    ENTERTAINMENT = "Развлечения"
    CLOTHING = "Одежда"
    UTILITIES = "Коммунальные услуги"
    EDUCATION = "Образование"
    GIFTS = "Подарки"
    OTHER_EXPENSE = "Прочие расходы"
    
    # Доходы
    SALARY = "Зарплата"
    BONUS = "Премия"
    INVESTMENT = "Инвестиции"
    GIFT_INCOME = "Подарок"
    OTHER_INCOME = "Прочие доходы"


