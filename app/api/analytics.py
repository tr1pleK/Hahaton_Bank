# app/api/analytics.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import case
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List

from app.database import get_db
from app.models.transaction import Transaction
from app.models.user import User
from app.models.category import TransactionCategory

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/balance-history")
def get_balance_history(email: str, db: Session = Depends(get_db)):
    """
    Возвращает историю баланса пользователя за последние 7 дней.
    Баланс на каждый день рассчитывается на основе текущего баланса и транзакций.
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    today = datetime.now().date()
    start_date = today - timedelta(days=6)
    all_dates = [start_date + timedelta(days=i) for i in range(7)]

    income_categories = [
        TransactionCategory.SALARY,
        TransactionCategory.BONUS,
        TransactionCategory.INVESTMENT,
        TransactionCategory.GIFT_INCOME,
        TransactionCategory.OTHER_INCOME
    ]

    # Получаем ВСЕ транзакции с start_date по today
    transactions = (
        db.query(Transaction.date, Transaction.category, Transaction.amount)
        .filter(Transaction.user_id == user.id)
        .filter(Transaction.date >= start_date)
        .filter(Transaction.date <= today)
        .all()
    )

    # Считаем дневные изменения
    daily_change = {date: 0.0 for date in all_dates}
    total_change_since_start = 0.0

    for tx in transactions:
        amount = float(tx.amount)
        signed_amount = amount if tx.category in income_categories else -amount
        if tx.date in daily_change:
            daily_change[tx.date] += signed_amount
        total_change_since_start += signed_amount

    # Текущий баланс (после всех транзакций до today включительно)
    current_balance = float(user.balance)

    # Баланс на начало периода (на дату start_date - 1)
    balance_before_period = current_balance - total_change_since_start

    # Строим историю: баланс на КОНЕЦ каждого дня
    history = []
    running_balance = balance_before_period

    for date in all_dates:
        running_balance += daily_change[date]
        history.append({
            "date": date.isoformat(),
            "balance": round(running_balance, 2)
        })

    return {
        "balance_history": history
    }