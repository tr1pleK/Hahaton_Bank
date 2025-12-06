# app/api/analytics.py
from calendar import monthrange
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import case
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Dict, List

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

def get_month_key(d: date) -> str:
    return d.strftime("%Y-%m")

@router.get("/expenses-last-3-months")
def get_expenses_last_3_months_by_month(email: str, db: Session = Depends(get_db)):
    """
    Возвращает расходы по категориям за каждый из трёх последних календарных месяцев.
    Формат:
    {
      "2025-12": [{"category": "TRANSPORT", "amount": 1200.5}, ...],
      "2025-11": [...],
      "2025-10": [...]
    }
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    now = datetime.now().date()

    # Определяем границы трёх месяцев
    month_ranges = []
    for i in range(3):
        if now.month - i > 0:
            year = now.year
            month = now.month - i
        else:
            year = now.year - 1
            month = 12 + (now.month - i)
        start = date(year, month, 1)
        end_day = monthrange(year, month)[1]
        end = date(year, month, end_day)
        if i == 0:
            end = now  # текущий месяц — только до сегодня
        month_ranges.append((start, end, get_month_key(start)))

    # Доходные категории
    income_categories = [
        TransactionCategory.SALARY,
        TransactionCategory.BONUS,
        TransactionCategory.INVESTMENT,
        TransactionCategory.GIFT_INCOME,
        TransactionCategory.OTHER_INCOME,
    ]

    # Инициализируем результат
    result = {month_key: [] for _, _, month_key in month_ranges}

    # Получаем все транзакции за нужный период
    earliest_start = min(r[0] for r in month_ranges)
    transactions = (
        db.query(Transaction.date, Transaction.category, Transaction.amount)
        .filter(Transaction.user_id == user.id)
        .filter(Transaction.date >= earliest_start)
        .filter(Transaction.date <= now)
        .filter(~Transaction.category.in_(income_categories))
        .all()
    )

    # Группируем
    monthly_data: Dict[str, Dict[str, float]] = {
        month_key: {} for month_key in result.keys()
    }

    for tx in transactions:
        tx_month = get_month_key(tx.date)
        if tx_month not in monthly_data:
            continue

        cat = tx.category.value if hasattr(tx.category, 'value') else tx.category
        amount = float(tx.amount)

        if cat in monthly_data[tx_month]:
            monthly_data[tx_month][cat] += amount
        else:
            monthly_data[tx_month][cat] = amount

    # Преобразуем в список объектов
    for month_key, cat_amounts in monthly_data.items():
        result[month_key] = [
            {"category": cat, "amount": round(amount, 2)}
            for cat, amount in cat_amounts.items()
        ]

    # Обеспечиваем порядок: от старого к новому (октябрь → ноябрь → декабрь)
    ordered_result = {
        month_key: result[month_key]
        for _, _, month_key in reversed(month_ranges)
    }

    return ordered_result