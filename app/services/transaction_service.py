"""Сервис для работы с транзакциями"""
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from typing import Optional, List
from datetime import date

from app.models.transaction import Transaction
from app.models.category import TransactionCategory
from app.schemas.transaction import TransactionCreate, TransactionUpdate
from app.ml.categorizer import categorize_transaction


def create_transaction(db: Session, user_id: int, transaction_data: TransactionCreate) -> Transaction:
    """Создает новую транзакцию"""
    category = None
    if transaction_data.category:
        try:
            category = TransactionCategory(transaction_data.category)
        except ValueError:
            category = None
    
    if not category:
        category = categorize_transaction(
            description=transaction_data.description or "",
            amount=float(transaction_data.amount)
        )
    
    transaction = Transaction(
        user_id=user_id,
        amount=transaction_data.amount,
        description=transaction_data.description,
        date=transaction_data.date,
        category=category
    )
    
    db.add(transaction)
    db.commit()
    db.refresh(transaction)
    return transaction


def get_transaction(db: Session, transaction_id: int, user_id: int) -> Optional[Transaction]:
    """Получает транзакцию по ID"""
    return db.query(Transaction).filter(
        and_(Transaction.id == transaction_id, Transaction.user_id == user_id)
    ).first()


def get_transactions(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 100,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    category: Optional[TransactionCategory] = None,
    is_income: Optional[bool] = None
) -> tuple[List[Transaction], int]:
    """Получает список транзакций"""
    query = db.query(Transaction).filter(Transaction.user_id == user_id)
    
    if start_date:
        query = query.filter(Transaction.date >= start_date)
    if end_date:
        query = query.filter(Transaction.date <= end_date)
    if category:
        query = query.filter(Transaction.category == category)
    if is_income is not None:
        income_cats = [
            TransactionCategory.SALARY, TransactionCategory.BONUS,
            TransactionCategory.INVESTMENT, TransactionCategory.GIFT_INCOME,
            TransactionCategory.OTHER_INCOME
        ]
        if is_income:
            query = query.filter(Transaction.category.in_(income_cats))
        else:
            expense_cats = [
                TransactionCategory.PRODUCTS, TransactionCategory.TRANSPORT,
                TransactionCategory.CAFE, TransactionCategory.HEALTH,
                TransactionCategory.ENTERTAINMENT, TransactionCategory.CLOTHING,
                TransactionCategory.UTILITIES, TransactionCategory.EDUCATION,
                TransactionCategory.GIFTS, TransactionCategory.OTHER_EXPENSE
            ]
            query = query.filter(Transaction.category.in_(expense_cats))
    
    total = query.count()
    transactions = query.order_by(desc(Transaction.date)).offset(skip).limit(limit).all()
    return transactions, total


def update_transaction(
    db: Session, transaction_id: int, user_id: int, transaction_data: TransactionUpdate
) -> Optional[Transaction]:
    """Обновляет транзакцию"""
    transaction = get_transaction(db, transaction_id, user_id)
    if not transaction:
        return None
    
    if transaction_data.amount is not None:
        transaction.amount = transaction_data.amount
    if transaction_data.description is not None:
        transaction.description = transaction_data.description
    if transaction_data.date is not None:
        transaction.date = transaction_data.date
    if transaction_data.category is not None:
        try:
            transaction.category = TransactionCategory(transaction_data.category)
        except ValueError:
            raise ValueError(f"Неверная категория: {transaction_data.category}")
    
    db.commit()
    db.refresh(transaction)
    return transaction


def delete_transaction(db: Session, transaction_id: int, user_id: int) -> bool:
    """Удаляет транзакцию"""
    transaction = get_transaction(db, transaction_id, user_id)
    if not transaction:
        return False
    db.delete(transaction)
    db.commit()
    return True
