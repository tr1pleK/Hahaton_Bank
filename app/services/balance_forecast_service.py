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
    Определение зарплаты и аренды по шаблонам на основе реальных данных.
    
    Минимальные требования для определения паттернов:
    - Для зарплаты: минимум 2 транзакции с крупными поступлениями (>20000) в период 20-30 числа
    - Для аренды: минимум 2 транзакции с крупными расходами (>2000) в период 1-10 числа
    
    Returns:
        tuple: (salary_day, salary_amount, rent_day, rent_amount, salary_detected, rent_detected)
        где salary_detected и rent_detected - флаги, указывающие, были ли паттерны определены на основе данных
    """
    # Минимальное количество транзакций для надежного определения паттернов
    MIN_TRANSACTIONS_FOR_PATTERN = 5
    MIN_SALARY_OCCURRENCES = 2  # Минимум 2 зарплаты для определения паттерна
    MIN_RENT_OCCURRENCES = 2    # Минимум 2 аренды для определения паттерна
    
    # Если данных недостаточно, возвращаем None для неопределенных значений
    if len(df) < MIN_TRANSACTIONS_FOR_PATTERN:
        return None, None, None, None, False, False
    
    salary_detected = False
    salary_day = None
    salary_amount = None
    
    # Определение зарплаты: ищем регулярные крупные поступления
    # Ищем поступления больше 20000 (более гибкий порог)
    large_deposits = df[df['Deposit'] > 20000].copy()
    
    if len(large_deposits) >= MIN_SALARY_OCCURRENCES:
        # Группируем по дням месяца и ищем наиболее частый день
        large_deposits['day_of_month'] = large_deposits['Date'].dt.day
        # Ищем дни в диапазоне 20-30 (типичный период зарплаты)
        salary_candidates = large_deposits[
            (large_deposits['day_of_month'] >= 20) & 
            (large_deposits['day_of_month'] <= 30)
        ]
        
        if len(salary_candidates) >= MIN_SALARY_OCCURRENCES:
            # Находим наиболее частый день зарплаты
            day_counts = salary_candidates['day_of_month'].value_counts()
            if len(day_counts) > 0:
                salary_day = int(day_counts.index[0])
                # Берем медиану сумм зарплат для этого дня
                salary_for_day = salary_candidates[
                    salary_candidates['day_of_month'] == salary_day
                ]['Deposit']
                if len(salary_for_day) > 0:
                    salary_amount = float(salary_for_day.median())
                    salary_detected = True
    
    # Если не нашли в диапазоне 20-30, пробуем более широкий поиск
    if not salary_detected and len(large_deposits) >= MIN_SALARY_OCCURRENCES:
        # Ищем любые регулярные крупные поступления
        large_deposits['day_of_month'] = large_deposits['Date'].dt.day
        day_counts = large_deposits['day_of_month'].value_counts()
        # Ищем день, который встречается минимум MIN_SALARY_OCCURRENCES раз
        frequent_days = day_counts[day_counts >= MIN_SALARY_OCCURRENCES]
        if len(frequent_days) > 0:
            salary_day = int(frequent_days.index[0])
            salary_for_day = large_deposits[
                large_deposits['day_of_month'] == salary_day
            ]['Deposit']
            if len(salary_for_day) > 0:
                salary_amount = float(salary_for_day.median())
                salary_detected = True
    
    rent_detected = False
    rent_day = None
    rent_amount = None
    
    # Определение аренды: ищем регулярные крупные расходы в начале месяца
    # Вариант 1: Транзакции с категорией Rent
    rent_by_category = df[df['Category'] == 'Rent'].copy()
    
    if len(rent_by_category) >= MIN_RENT_OCCURRENCES:
        rent_by_category['day_of_month'] = rent_by_category['Date'].dt.day
        day_counts = rent_by_category['day_of_month'].value_counts()
        frequent_days = day_counts[day_counts >= MIN_RENT_OCCURRENCES]
        if len(frequent_days) > 0:
            rent_day = int(frequent_days.index[0])
            rent_for_day = rent_by_category[
                rent_by_category['day_of_month'] == rent_day
            ]['Withdrawal']
            if len(rent_for_day) > 0:
                rent_amount = float(rent_for_day.median())
                rent_detected = True
    
    # Вариант 2: Крупные расходы в начале месяца (1-10 число)
    if not rent_detected:
        early_month_large = df[
            (df['Withdrawal'] > 2000) & 
            (df['Date'].dt.day >= 1) & 
            (df['Date'].dt.day <= 10)
        ].copy()
        
        if len(early_month_large) >= MIN_RENT_OCCURRENCES:
            early_month_large['day_of_month'] = early_month_large['Date'].dt.day
            day_counts = early_month_large['day_of_month'].value_counts()
            frequent_days = day_counts[day_counts >= MIN_RENT_OCCURRENCES]
            if len(frequent_days) > 0:
                rent_day = int(frequent_days.index[0])
                rent_for_day = early_month_large[
                    early_month_large['day_of_month'] == rent_day
                ]['Withdrawal']
                if len(rent_for_day) > 0:
                    rent_amount = float(rent_for_day.median())
                    rent_detected = True
    
    return salary_day, salary_amount, rent_day, rent_amount, salary_detected, rent_detected


def compute_spending_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Вычисление статистики расходов.
    
    Для новых пользователей (без транзакций) возвращает консервативные значения.
    """
    if len(df) == 0:
        return {
            'avg_daily_spending': 0.0,  # Не делаем предположений о расходах
            'avg_daily_income': 0.0,
            'total_misc_withdrawal': 0.0,
            'total_food_withdrawal': 0.0,
        }
    
    # Убеждаемся, что NetFlow существует (создается в load_transactions_from_db)
    if 'NetFlow' not in df.columns:
        df['NetFlow'] = df['Deposit'] - df['Withdrawal']
    
    # Исключаем регулярные платежи (Rent, Salary) для расчета переменных расходов
    variable_tx = df[~df['Category'].isin(['Rent', 'Salary'])].copy()
    
    # Средние ежедневные расходы (только если есть данные)
    spending_tx = variable_tx[variable_tx['NetFlow'] < 0]
    if len(spending_tx) > 0:
        # Вычисляем средний расход за день на основе реальных данных
        spending_tx = spending_tx.copy()
        spending_tx['days_diff'] = spending_tx['Date'].diff().dt.days.fillna(1)
        total_spending = abs(spending_tx['NetFlow'].sum())
        total_days = spending_tx['days_diff'].sum()
        avg_daily_spending = -float(total_spending / total_days) if total_days > 0 else 0.0
    else:
        avg_daily_spending = 0.0
    
    # Средние ежедневные доходы (только если есть данные)
    income_tx = variable_tx[variable_tx['NetFlow'] > 0]
    if len(income_tx) > 0:
        income_tx = income_tx.copy()
        income_tx['days_diff'] = income_tx['Date'].diff().dt.days.fillna(1)
        total_income = income_tx['NetFlow'].sum()
        total_days = income_tx['days_diff'].sum()
        avg_daily_income = float(total_income / total_days) if total_days > 0 else 0.0
    else:
        avg_daily_income = 0.0
    
    total_misc_withdrawal = float(variable_tx[variable_tx['Category'] == 'Misc']['Withdrawal'].sum())
    total_food_withdrawal = float(variable_tx[variable_tx['Category'] == 'Food']['Withdrawal'].sum())
    
    return {
        'avg_daily_spending': avg_daily_spending,
        'avg_daily_income': avg_daily_income,
        'total_misc_withdrawal': total_misc_withdrawal,
        'total_food_withdrawal': total_food_withdrawal,
    }


def assess_budget_stability(
    salary_amount: Optional[float], 
    rent_amount: Optional[float], 
    avg_daily_spending: float
) -> float:
    """
    Оценка устойчивости бюджета.
    
    Если зарплата или аренда не определены, возвращает 0 (недостаточно данных).
    """
    # Если нет данных о зарплате или аренде, не можем оценить устойчивость
    if salary_amount is None or rent_amount is None:
        return 0.0
    
    net_income = salary_amount - rent_amount
    avg_var_spend = -(avg_daily_spending * 30) if avg_daily_spending < 0 else 0.1
    return net_income / avg_var_spend if avg_var_spend > 0 else 0.0


def estimate_financial_pillow(df: pd.DataFrame, salary_day: Optional[int]) -> float:
    """
    Оценка финансовой подушки.
    
    Для новых пользователей возвращает 0.0 (недостаточно данных).
    """
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
        # Если нет данных о зарплате, используем минимальный баланс за последние дни месяца
        if salary_day is not None:
            last_days = df[df['Date'].dt.day >= salary_day - 5]
        else:
            last_days = df[df['Date'].dt.day >= 20]
        
        if not last_days.empty:
            return float(last_days['Balance'].min())
        else:
            return float(df['Balance'].min())


def forecast_to_month_end(
    current_date: date,
    current_balance: float,
    salary_day: Optional[int],
    salary_amount: Optional[float],
    rent_day: Optional[int],
    rent_amount: Optional[float],
    avg_daily_spending: float,
    avg_daily_income: float
) -> pd.DataFrame:
    """
    Прогноз баланса до конца текущего месяца.
    
    Args:
        current_date: Текущая дата (вчерашний день)
        current_balance: Текущий баланс
        salary_day: День зарплаты (None если не определен)
        salary_amount: Сумма зарплаты (None если не определен)
        rent_day: День аренды (None если не определен)
        rent_amount: Сумма аренды (None если не определен)
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
        # Определяем доходы на этот день
        if salary_day is not None and salary_amount is not None and day.day == salary_day:
            income = salary_amount
        else:
            income = max(avg_daily_income, 0)
        
        # Определяем расходы на этот день
        if rent_day is not None and rent_amount is not None and day.day == rent_day:
            expense = rent_amount
        else:
            expense = max(-avg_daily_spending, 0)
        
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
    salary_day, salary_amount, rent_day, rent_amount, salary_detected, rent_detected = detect_fixed_events(df)
    
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
    
    # Добавляем специальные рекомендации для новых пользователей
    if not salary_detected or not rent_detected:
        if len(df) == 0:
            recommendations.insert(0, "ℹ️ У вас пока нет транзакций. Прогноз будет доступен после добавления нескольких транзакций.")
        elif len(df) < 5:
            recommendations.insert(0, "ℹ️ У вас недостаточно транзакций для точного прогноза. Добавьте больше транзакций для улучшения точности.")
        if not salary_detected:
            recommendations.append("💡 Система пока не определила день зарплаты. После добавления нескольких крупных поступлений прогноз станет точнее.")
        if not rent_detected:
            recommendations.append("💡 Система пока не определила день аренды. После добавления нескольких крупных расходов в начале месяца прогноз станет точнее.")
    
    # Прогноз на конец месяца
    end_of_month_balance = forecast_df['PredictedBalance'].iloc[-1] if len(forecast_df) > 0 else current_balance
    
    return {
        'summary': {
            'last_date': last_date.strftime('%Y-%m-%d'),
            'last_balance': round(last_balance, 2),
            'current_balance': round(current_balance, 2),
            'salary_day': salary_day if salary_detected else None,
            'salary_amount': round(salary_amount, 2) if salary_amount is not None else None,
            'salary_detected': salary_detected,
            'rent_day': rent_day if rent_detected else None,
            'rent_amount': round(rent_amount, 2) if rent_amount is not None else None,
            'rent_detected': rent_detected,
            'budget_stability': round(stability, 2),
            'financial_pillow': round(pillow, 2),
            'avg_daily_spending': round(-stats['avg_daily_spending'], 2),
            'forecast_end_of_month': round(end_of_month_balance, 2),
            'forecast_date': forecast_df['Date'].iloc[-1] if len(forecast_df) > 0 else None
        },
        'forecast_daily': forecast_df.to_dict('records'),
        'recommendations': recommendations
    }

