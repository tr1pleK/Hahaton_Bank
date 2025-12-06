import pandas as pd
import numpy as np
import joblib
import os
from datetime import date, timedelta
from sklearn.metrics import f1_score
import lightgbm as lgb
from sqlalchemy.orm import Session
from pathlib import Path

# Импорты для работы с БД
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.database import SessionLocal
from app.models.transaction import Transaction
from app.models.category import TransactionCategory

# === Функция извлечения признаков (должна быть идентична той, что в обучении) ===
def extract_features(df):
    df = df.sort_values("Date").reset_index(drop=True)
    df["is_withdrawal"] = (df["Withdrawal"] > 0).astype(int)
    df["is_deposit"] = (df["Deposit"] > 0).astype(int)
    df["amount"] = df["Withdrawal"] + df["Deposit"]
    df["net_flow"] = df["Deposit"] - df["Withdrawal"]
    df["balance_before"] = df["Balance"] + df["Withdrawal"] - df["Deposit"]

    df["day_of_month"] = df["Date"].dt.day
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_month_start"] = (df["day_of_month"] <= 10).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 24).astype(int)

    df["is_salary_like"] = (df["Deposit"] == 34800).astype(int)
    df["is_rent_like"] = (
        (df["Withdrawal"] >= 3900) & 
        (df["Withdrawal"] <= 7500) & 
        (df["day_of_month"] <= 6)
    ).astype(int)

    salary_dates = df[df["Deposit"] == 34800]["Date"].tolist()
    df["days_since_last_salary"] = np.nan
    for i, row in df.iterrows():
        past_salaries = [d for d in salary_dates if d <= row["Date"]]
        if past_salaries:
            last_salary = max(past_salaries)
            df.at[i, "days_since_last_salary"] = (row["Date"] - last_salary).days

    df["days_since_last_txn"] = df["Date"].diff().dt.days.fillna(0)
    return df

# Маппинг категорий системы в категории модели
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

def load_transactions_from_db(db: Session, days_back: int = 7) -> pd.DataFrame:
    """
    Загрузка транзакций из БД за последние N дней для всех пользователей.
    
    Args:
        db: Сессия базы данных
        days_back: Количество дней назад для выборки (по умолчанию 7)
        
    Returns:
        DataFrame с колонками: Date, Category, RefNo, Withdrawal, Deposit, Balance
    """
    # Определяем диапазон дат
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"📅 Загрузка транзакций за период: {start_date} - {end_date}")
    
    # Получаем все транзакции за период
    transactions = db.query(Transaction).filter(
        Transaction.date >= start_date,
        Transaction.date <= end_date
    ).order_by(Transaction.date).all()
    
    if not transactions:
        print("⚠️ Нет транзакций за указанный период")
        return pd.DataFrame(columns=['Date', 'Category', 'RefNo', 'Withdrawal', 'Deposit', 'Balance'])
    
    print(f"✅ Найдено транзакций: {len(transactions)}")
    
    # Преобразуем транзакции в список словарей
    data = []
    for txn in transactions:
        # Преобразуем категорию системы в категорию модели
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
    
    # Преобразуем Date в datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Вычисляем баланс последовательно (группируем по пользователям)
    # Для упрощения вычисляем общий баланс как накопительную сумму
    df['NetFlow'] = df['Deposit'] - df['Withdrawal']
    df['Balance'] = df['NetFlow'].cumsum()
    
    return df


# === Основная функция дообучения ===
def weekly_retrain():
    """Дообучение модели на основе транзакций из БД за последнюю неделю"""
    # Пути
    MODEL_PATH = Path(__file__).parent / "classifier_v2.pkl"
    FEATURE_COLS = [
        "is_withdrawal", "is_deposit", "amount", "net_flow", "balance_before",
        "day_of_month", "day_of_week", "is_month_start", "is_month_end",
        "is_salary_like", "is_rent_like", "days_since_last_salary", "days_since_last_txn",
        "Withdrawal", "Deposit"
    ]
    VALID_CATS = ["Food", "Misc", "Rent", "Salary", "Shopping"]

    # 1. Загрузка данных из БД
    db = SessionLocal()
    try:
        df_new = load_transactions_from_db(db, days_back=7)
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных из БД: {e}")
        import traceback
        traceback.print_exc()
        db.close()
        return
    finally:
        db.close()
    
    # 2. Проверка наличия данных
    if len(df_new) == 0:
        print("⚠️ Нет новых данных для дообучения.")
        return
    
    if len(df_new) < 10:
        print(f"⚠️ Слишком мало новых данных ({len(df_new)} строк < 10) — пропускаем дообучение.")
        return

    # 3. Фильтрация категорий
    df_new = df_new[df_new["Category"].isin(VALID_CATS)].copy().reset_index(drop=True)
    
    if len(df_new) == 0:
        print("⚠️ Нет валидных категорий после фильтрации.")
        return

    # 4. Извлечение признаков
    print("🔧 Извлечение признаков...")
    df_new = extract_features(df_new)
    X_new = df_new[FEATURE_COLS].fillna(-1)
    y_new = df_new["Category"]
    
    print(f"📊 Данных для дообучения: {len(X_new)}")
    print(f"📊 Распределение категорий:\n{y_new.value_counts()}")

    # 5. Загрузка текущей модели
    if not MODEL_PATH.exists():
        print(f"❌ Модель не найдена по пути: {MODEL_PATH}")
        return
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Модель загружена из {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. Дообучение (boosting from existing model)
    print("🔄 Начало дообучения модели...")
    model_new = lgb.LGBMClassifier(
        boosting_type='gbdt',
        n_estimators=50,  # Небольшое количество новых деревьев
        num_leaves=15,
        learning_rate=0.02,  # Меньше — осторожнее
        min_data_in_leaf=10,
        lambda_l1=0.1,
        lambda_l2=0.1,
        random_state=42,
        class_weight="balanced"
    )

    # Важно: передаём init_model для дообучения
    try:
        model_new.fit(
            X_new, y_new,
            init_model=model,
            eval_set=[(X_new, y_new)],
            verbose=10
        )
    except Exception as e:
        print(f"❌ Ошибка при дообучении: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. Оценка (опционально)
    y_pred = model_new.predict(X_new)
    new_f1 = f1_score(y_new, y_pred, average="weighted")
    print(f"✅ F1 на новых данных после дообучения: {new_f1:.4f}")

    # 8. Сохранение новой модели
    try:
        joblib.dump(model_new, MODEL_PATH)
        print(f"✅ Модель успешно дообучена и сохранена в {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Ошибка при сохранении модели: {e}")
        import traceback
        traceback.print_exc()
        return
