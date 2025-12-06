"""Сервис для прогнозирования баланса на основе алгоритма"""
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional
from calendar import monthrange

from app.models.transaction import Transaction
from app.models.category import TransactionCategory
from app.models.user import User

# Маппинг категорий системы в категории алгоритма
CATEGORY_MAPPING = {
    TransactionCategory.PRODUCTS: 'Food',
    TransactionCategory.TRANSPORT: 'Misc',
    TransactionCategory.CAFE: 'Food',
    TransactionCategory.HEALTH: 'Misc',
    TransactionCategory.ENTERTAINMENT: 'Misc',
    TransactionCategory.CLOTHING: 'Shopping',
    TransactionCategory.UTILITIES: 'Rent',
    TransactionCategory.EDUCATION: 'Misc',
    TransactionCategory.GIFTS: 'Misc',
    TransactionCategory.OTHER_EXPENSE: 'Misc',
    TransactionCategory.SALARY: 'Salary',
    TransactionCategory.BONUS: 'Salary',
    TransactionCategory.INVESTMENT: 'Misc',
    TransactionCategory.GIFT_INCOME: 'Misc',
    TransactionCategory.OTHER_INCOME: 'Misc',
}


def load_transactions_from_db(db: Session, user_id: int) -> pd.DataFrame:
    """
    Загрузка и очистка данных транзакций из БД для пользователя.
    
    Args:
        db: Сессия базы данных
        user_id: ID пользователя
        
    Returns:
        DataFrame с колонками: Date, Category, RefNo, Withdrawal, Deposit, Balance
    """
    # Определяем диапазон дат: максимум 1 год назад, до вчерашнего дня
    today = date.today()
    yesterday = today - timedelta(days=1)
    one_year_ago = today - timedelta(days=365)
    
    # Получаем все транзакции пользователя
    transactions = db.query(Transaction).filter(
        and_(
            Transaction.user_id == user_id,
            Transaction.date >= one_year_ago,
            Transaction.date <= yesterday
        )
    ).order_by(Transaction.date).all()
    
    if not transactions:
        # Возвращаем пустой DataFrame с правильными колонками
        return pd.DataFrame(columns=['Date', 'Category', 'RefNo', 'Withdrawal', 'Deposit', 'Balance'])
    
    # Получаем текущий баланс пользователя
    user = db.query(User).filter(User.id == user_id).first()
    current_balance = float(user.balance) if user else 0.0
    
    # Преобразуем транзакции в список словарей
    data = []
    for txn in transactions:
        # Преобразуем категорию системы в категорию алгоритма
        category = CATEGORY_MAPPING.get(txn.category, 'Misc')
        
        # Определяем Withdrawal и Deposit
        withdrawal = 0.0
        deposit = 0.0
        
        if txn.is_income:
            deposit = float(txn.amount)
        else:
            withdrawal = float(txn.amount)
        
        data.append({
            'Date': txn.date,
            'Category': category,
            'RefNo': txn.description or '',
            'Withdrawal': withdrawal,
            'Deposit': deposit,
            'Balance': 0.0  # Будет вычислен позже
        })
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        return df
    
    # Сортируем по дате
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Преобразуем Date в datetime, если это еще не сделано
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Вычисляем баланс последовательно
    # Текущий баланс = начальный баланс + сумма всех транзакций
    # Начальный баланс = текущий баланс - сумма всех транзакций
    total_net_flow = (df['Deposit'] - df['Withdrawal']).sum()
    start_balance = current_balance - total_net_flow
    
    # Вычисляем баланс последовательно от начального баланса
    df['NetFlow'] = df['Deposit'] - df['Withdrawal']
    df['Balance'] = start_balance + df['NetFlow'].cumsum()
    
    return df


def detect_fixed_events(df: pd.DataFrame) -> tuple:
    """
    Определение зарплаты и аренды по шаблонам.
    
    Returns:
        tuple: (salary_day, salary_amount, rent_day, rent_amount)
    """
    if len(df) == 0:
        return 25, 34800, 1, 6500
    
    salary_mask = (df['Deposit'] > 30000) & (df['Date'].dt.day.between(23, 27))
    salary_day = int(df[salary_mask]['Date'].dt.day.mode().iloc[0]) if not df[salary_mask].empty else 25
    salary_amount = float(df[salary_mask]['Deposit'].median()) if not df[salary_mask].empty else 34800

    rent_mask = (df['Category'] == 'Rent') | ((df['Withdrawal'] > 3000) & (df['Date'].dt.day <= 6))
    rent_day = int(df[rent_mask]['Date'].dt.day.mode().iloc[0]) if not df[rent_mask].empty else 1
    rent_amount = float(df[rent_mask]['Withdrawal'].median()) if not df[rent_mask].empty else 6500

    return salary_day, salary_amount, rent_day, rent_amount


def compute_spending_stats(df: pd.DataFrame) -> Dict[str, float]:
    """Вычисление статистики расходов."""
    if len(df) == 0:
        return {
            'avg_daily_spending': -500.0,
            'avg_daily_income': 0.0,
            'total_misc_withdrawal': 0.0,
            'total_food_withdrawal': 0.0,
        }
    
    variable_tx = df[~df['Category'].isin(['Rent', 'Salary'])].copy()
    avg_daily_spending = float(variable_tx[variable_tx['NetFlow'] < 0]['NetFlow'].mean()) if len(variable_tx[variable_tx['NetFlow'] < 0]) > 0 else -500.0
    avg_daily_income = float(variable_tx[variable_tx['NetFlow'] > 0]['NetFlow'].mean()) if len(variable_tx[variable_tx['NetFlow'] > 0]) > 0 else 0.0
    total_misc_withdrawal = float(variable_tx[variable_tx['Category'] == 'Misc']['Withdrawal'].sum())
    total_food_withdrawal = float(variable_tx[variable_tx['Category'] == 'Food']['Withdrawal'].sum())
    
    return {
        'avg_daily_spending': avg_daily_spending,
        'avg_daily_income': avg_daily_income,
        'total_misc_withdrawal': total_misc_withdrawal,
        'total_food_withdrawal': total_food_withdrawal,
    }


def assess_budget_stability(salary_amount: float, rent_amount: float, avg_daily_spending: float) -> float:
    """Оценка устойчивости бюджета."""
    net_income = salary_amount - rent_amount
    avg_var_spend = -(avg_daily_spending * 30) if avg_daily_spending < 0 else 0.1
    return net_income / avg_var_spend if avg_var_spend > 0 else 0.0


def estimate_financial_pillow(df: pd.DataFrame, salary_day: int) -> float:
    """Оценка финансовой подушки."""
    if len(df) == 0:
        return 0.0
    
    df = df.copy()
    df['Month'] = df['Date'].dt.to_period('M')
    min_balances = []
    
    for _, group in df.groupby('Month'):
        salary_dates = group[group['Category'] == 'Salary']['Date']
        if not salary_dates.empty:
            sal_date = salary_dates.iloc[0]
            window = group[(group['Date'] >= sal_date - pd.Timedelta(days=5)) & (group['Date'] < sal_date)]
            if not window.empty:
                min_balances.append(float(window['Balance'].min()))
    
    if min_balances:
        return float(pd.Series(min_balances).median())
    else:
        last_days = df[df['Date'].dt.day >= 25]
        if not last_days.empty:
            return float(last_days['Balance'].min())
        else:
            return float(df['Balance'].min())


def forecast_to_month_end(
    current_date: date,
    current_balance: float,
    salary_day: int,
    salary_amount: float,
    rent_day: int,
    rent_amount: float,
    avg_daily_spending: float,
    avg_daily_income: float
) -> pd.DataFrame:
    """
    Прогноз баланса до конца текущего месяца.
    
    Args:
        current_date: Текущая дата (вчерашний день)
        current_balance: Текущий баланс
        salary_day: День зарплаты
        salary_amount: Сумма зарплаты
        rent_day: День аренды
        rent_amount: Сумма аренды
        avg_daily_spending: Средние ежедневные расходы
        avg_daily_income: Средние ежедневные доходы
        
    Returns:
        DataFrame с прогнозом на каждый день до конца месяца
    """
    # Определяем последний день текущего месяца
    last_day_of_month = monthrange(current_date.year, current_date.month)[1]
    target_date = date(current_date.year, current_date.month, last_day_of_month)
    
    forecast = []
    balance = current_balance
    day = current_date + timedelta(days=1)  # Начинаем с сегодняшнего дня
    
    while day <= target_date:
        # Определяем доходы и расходы на этот день
        income = salary_amount if day.day == salary_day else max(avg_daily_income, 0)
        expense = rent_amount if day.day == rent_day else max(-avg_daily_spending, 0)
        
        balance += income - expense
        
        forecast.append({
            'Date': day.strftime('%Y-%m-%d'),
            'PredictedBalance': round(balance, 2),
            'Income': round(income, 2),
            'Expense': round(expense, 2)
        })
        
        day += timedelta(days=1)
    
    return pd.DataFrame(forecast)


def generate_recommendations(
    financial_pillow: float,
    budget_stability: float,
    total_misc_withdrawal: float,
    total_food_withdrawal: float,
    df: pd.DataFrame
) -> list:
    """Генерация персонализированных рекомендаций."""
    recs = []
    
    if financial_pillow < 10000:
        recs.append("⚠️ Ваша финансовая подушка меньше 10 000 руб. Рекомендуем сократить расходы в категории Misc на 15–20%.")
    
    if budget_stability < 1.2:
        recs.append("❗ Бюджет неустойчив: расходы близки к доходам. Избегайте крупных покупок за неделю до зарплаты.")
    
    if total_misc_withdrawal > total_food_withdrawal:
        recs.append("🔍 Вы тратите больше на Misc, чем на Food. Рассмотрите возможность перераспределения этих расходов.")
    
    if not recs:
        recs.append("✅ Ваш бюджет в хорошей форме! Поддерживайте текущие привычки.")
    
    return recs


def get_balance_forecast(db: Session, user_id: int) -> Dict[str, Any]:
    """
    Получение прогноза баланса на конец месяца для пользователя.
    
    Args:
        db: Сессия базы данных
        user_id: ID пользователя
        
    Returns:
        Словарь с прогнозом и рекомендациями
    """
    # Загружаем транзакции из БД
    df = load_transactions_from_db(db, user_id)
    
    # Получаем текущий баланс пользователя
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise ValueError(f"Пользователь с ID {user_id} не найден")
    
    current_balance = float(user.balance)
    
    # Определяем вчерашний день (последний день с данными)
    yesterday = date.today() - timedelta(days=1)
    
    # Если есть транзакции, используем дату последней транзакции
    if len(df) > 0:
        last_date_pd = df['Date'].iloc[-1]
        # Преобразуем pandas Timestamp в date
        if isinstance(last_date_pd, pd.Timestamp):
            last_date = last_date_pd.date()
        elif hasattr(last_date_pd, 'date'):
            last_date = last_date_pd.date()
        else:
            last_date = yesterday
        last_balance = float(df['Balance'].iloc[-1])
    else:
        last_date = yesterday
        last_balance = current_balance
    
    # Определяем фиксированные события
    salary_day, salary_amount, rent_day, rent_amount = detect_fixed_events(df)
    
    # Вычисляем статистику расходов
    stats = compute_spending_stats(df)
    
    # Оценка устойчивости бюджета
    stability = assess_budget_stability(salary_amount, rent_amount, stats['avg_daily_spending'])
    
    # Оценка финансовой подушки
    pillow = estimate_financial_pillow(df, salary_day)
    
    # Прогноз до конца месяца
    forecast_df = forecast_to_month_end(
        last_date,
        last_balance,
        salary_day,
        salary_amount,
        rent_day,
        rent_amount,
        stats['avg_daily_spending'],
        stats['avg_daily_income']
    )
    
    # Генерация рекомендаций
    recommendations = generate_recommendations(
        pillow,
        stability,
        stats['total_misc_withdrawal'],
        stats['total_food_withdrawal'],
        df
    )
    
    # Прогноз на конец месяца
    end_of_month_balance = forecast_df['PredictedBalance'].iloc[-1] if len(forecast_df) > 0 else current_balance
    
    return {
        'summary': {
            'last_date': last_date.strftime('%Y-%m-%d'),
            'last_balance': round(last_balance, 2),
            'current_balance': round(current_balance, 2),
            'salary_day': salary_day,
            'salary_amount': round(salary_amount, 2),
            'rent_day': rent_day,
            'rent_amount': round(rent_amount, 2),
            'budget_stability': round(stability, 2),
            'financial_pillow': round(pillow, 2),
            'avg_daily_spending': round(-stats['avg_daily_spending'], 2),
            'forecast_end_of_month': round(end_of_month_balance, 2),
            'forecast_date': forecast_df['Date'].iloc[-1] if len(forecast_df) > 0 else None
        },
        'forecast_daily': forecast_df.to_dict('records'),
        'recommendations': recommendations
    }

