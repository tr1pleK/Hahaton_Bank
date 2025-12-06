"""Сервис для дообучения модели на основе данных пользователей"""
import pandas as pd
from sqlalchemy.orm import Session
from datetime import date, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

from app.models.transaction import Transaction
from app.models.category import TransactionCategory
from app.ml.transaction_classifier import TransactionClassifier, CATEGORY_MAPPING

# Обратный маппинг: категории системы -> категории модели
REVERSE_CATEGORY_MAPPING = {v: k for k, v in CATEGORY_MAPPING.items()}


def export_transactions_to_dataframe(
    db: Session,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Экспорт транзакций из БД в формат DataFrame для обучения
    
    Args:
        db: Сессия базы данных
        start_date: Начальная дата (опционально)
        end_date: Конечная дата (опционально)
        
    Returns:
        DataFrame с колонками: Date, Category, Withdrawal, Deposit, Balance
    """
    query = db.query(Transaction)
    
    if start_date:
        query = query.filter(Transaction.date >= start_date)
    if end_date:
        query = query.filter(Transaction.date <= end_date)
    
    transactions = query.order_by(Transaction.date).all()
    
    if not transactions:
        return pd.DataFrame(columns=["Date", "Category", "Withdrawal", "Deposit", "Balance"])
    
    # Преобразуем транзакции в список словарей
    data = []
    for txn in transactions:
        # Определяем, является ли транзакция доходом или расходом
        is_income = txn.is_income
        
        # Преобразуем категорию системы в категорию модели
        category_model = REVERSE_CATEGORY_MAPPING.get(txn.category, "Misc")
        
        # Если категория не в маппинге, пропускаем или используем Misc
        if category_model not in ["Food", "Misc", "Rent", "Salary", "Shopping"]:
            category_model = "Misc"
        
        # Вычисляем Withdrawal и Deposit
        withdrawal = 0.0
        deposit = 0.0
        
        if is_income:
            deposit = float(txn.amount)
        else:
            withdrawal = float(txn.amount)
        
        data.append({
            "Date": txn.date,
            "Category": category_model,
            "Withdrawal": withdrawal,
            "Deposit": deposit,
            "Balance": 0.0  # Будет вычислен позже
        })
    
    df = pd.DataFrame(data)
    
    # Сортируем по дате и вычисляем баланс последовательно
    if len(df) > 0:
        df = df.sort_values("Date").reset_index(drop=True)
        # Вычисляем баланс на основе транзакций (накопительный)
        # Начальный баланс = 0, затем добавляем/вычитаем транзакции
        df["Balance"] = (df["Deposit"] - df["Withdrawal"]).cumsum()
    
    return df


def retrain_model(
    db: Session,
    original_csv_path: Optional[str] = None,
    days_back: int = 7,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Дообучение модели на основе новых данных пользователей
    
    Args:
        db: Сессия базы данных
        original_csv_path: Путь к оригинальному CSV файлу для объединения данных
        days_back: Количество дней назад для выборки новых транзакций (по умолчанию 7)
        model_path: Путь к модели (опционально)
        
    Returns:
        Словарь с результатами дообучения
    """
    print("🔄 Начало дообучения модели...")
    
    # Определяем диапазон дат для новых транзакций
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"📅 Сбор данных за период: {start_date} - {end_date}")
    
    # Экспортируем транзакции из БД
    df_new = export_transactions_to_dataframe(db, start_date=start_date, end_date=end_date)
    
    if len(df_new) == 0:
        return {
            "success": False,
            "message": "Нет новых транзакций для дообучения",
            "new_transactions_count": 0
        }
    
    print(f"✅ Найдено новых транзакций: {len(df_new)}")
    print(f"📊 Распределение категорий:\n{df_new['Category'].value_counts()}")
    
    # Инициализируем классификатор
    classifier = TransactionClassifier(model_path=model_path)
    
    # Если модель не загружена, пытаемся загрузить
    if not classifier.is_trained:
        print("⚠️ Модель не загружена. Пытаемся загрузить существующую...")
        classifier.load_model()
    
    # Если оригинальный CSV не указан, пытаемся найти его в стандартном месте
    if original_csv_path is None:
        # Пробуем найти оригинальный CSV в разных местах
        possible_paths = [
            Path(__file__).parent.parent.parent / "ci_data.csv",
            Path(__file__).parent.parent.parent / "ml_models" / "ci_data.csv",
            Path(__file__).parent / "ci_data.csv",
        ]
        
        for path in possible_paths:
            if path.exists():
                original_csv_path = str(path)
                print(f"📂 Найден оригинальный CSV: {original_csv_path}")
                break
    
    # Обучаем модель
    try:
        metrics = classifier.train(df_new, original_csv_path=original_csv_path)
        
        return {
            "success": True,
            "message": "Модель успешно дообучена",
            "new_transactions_count": len(df_new),
            "metrics": metrics,
            "model_path": classifier.model_path
        }
    except Exception as e:
        print(f"❌ Ошибка при дообучении модели: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Ошибка при дообучении: {str(e)}",
            "new_transactions_count": len(df_new)
        }

