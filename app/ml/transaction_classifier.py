"""Классификатор транзакций на основе LightGBM"""
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score

from app.models.category import TransactionCategory

# Маппинг категорий из модели в категории системы
CATEGORY_MAPPING = {
    'Rent': TransactionCategory.UTILITIES,
    'Misc': TransactionCategory.OTHER_EXPENSE,
    'Food': TransactionCategory.PRODUCTS,
    'Salary': TransactionCategory.SALARY,
    'Shopping': TransactionCategory.CLOTHING,
    'Transport': TransactionCategory.TRANSPORT,
}

# Обратный маппинг для преобразования категорий системы в категории модели
REVERSE_CATEGORY_MAPPING = {v: k for k, v in CATEGORY_MAPPING.items()}

# Валидные категории модели
VALID_CATEGORIES = ["Food", "Misc", "Rent", "Salary", "Shopping"]


class TransactionClassifier:
    """Классификатор транзакций на основе LightGBM с fallback правилами"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация классификатора
        
        Args:
            model_path: Путь к сохраненной модели (опционально)
        """
        if model_path:
            self.model_path = model_path
        else:
            # Путь по умолчанию
            default_path = Path(__file__).parent.parent.parent / "ml_models" / "transaction_classifier.pkl"
            self.model_path = str(default_path)
        
        self.model = None
        self.is_trained = False
        
        # Загрузите модель, если она существует
        if os.path.exists(self.model_path):
            print(f"🔍 Найдена модель по пути: {self.model_path}")
            self.load_model()
            if self.is_trained:
                print(f"✅ Модель успешно загружена и готова к использованию")
        else:
            print(f"⚠️ Модель не найдена по пути: {self.model_path}")
            print(f"   Используется fallback классификация (вероятность всегда 0.5)")
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение признаков из DataFrame (как в cybergarden_ML.py)
        
        Args:
            df: DataFrame с колонками Date, Category (опционально), Withdrawal, Deposit, Balance
            
        Returns:
            DataFrame с извлеченными признаками
        """
        df = df.sort_values("Date").reset_index(drop=True)
        
        # Базовые признаки
        df["is_withdrawal"] = (df["Withdrawal"] > 0).astype(int)
        df["is_deposit"] = (df["Deposit"] > 0).astype(int)
        df["amount"] = df["Withdrawal"] + df["Deposit"]
        df["net_flow"] = df["Deposit"] - df["Withdrawal"]
        df["balance_before"] = df["Balance"] + df["Withdrawal"] - df["Deposit"]
        
        # Признаки даты
        df["day_of_month"] = df["Date"].dt.day
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["is_month_start"] = (df["day_of_month"] <= 10).astype(int)
        df["is_month_end"] = (df["day_of_month"] >= 24).astype(int)
        
        # Бизнес-правила
        df["is_salary_like"] = (df["Deposit"] == 34800).astype(int)
        df["is_rent_like"] = (
            (df["Withdrawal"] >= 3900) & 
            (df["Withdrawal"] <= 7500) & 
            (df["day_of_month"] <= 6)
        ).astype(int)
        
        # Days since last salary
        salary_dates = df[df["Deposit"] == 34800]["Date"].tolist()
        df["days_since_last_salary"] = np.nan
        for i, row in df.iterrows():
            past_salaries = [d for d in salary_dates if d <= row["Date"]]
            if past_salaries:
                last_salary = max(past_salaries)
                df.at[i, "days_since_last_salary"] = (row["Date"] - last_salary).days
        
        # Days since last transaction
        df["days_since_last_txn"] = df["Date"].diff().dt.days.fillna(0)
        
        return df
    
    def _get_feature_columns(self) -> List[str]:
        """Возвращает список колонок признаков"""
        return [
            "is_withdrawal", "is_deposit", "amount", "net_flow", "balance_before",
            "day_of_month", "day_of_week", "is_month_start", "is_month_end",
            "is_salary_like", "is_rent_like", "days_since_last_salary", "days_since_last_txn",
            "Withdrawal", "Deposit"
        ]
    
    def _apply_fallback_rules(self, X: pd.DataFrame, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применение fallback правил (как в test_classifier.py)
        
        Args:
            X: DataFrame с признаками
            y_pred_proba: Массив вероятностей предсказаний
            
        Returns:
            Tuple с предсказанными категориями и вероятностями
        """
        y_pred_proba = y_pred_proba.copy()
        categories = self.model.classes_
        
        for i in range(len(X)):
            row = X.iloc[i]
            
            # Rule 1: Salary - Deposit == 34800 и дата 24–26
            if row["is_salary_like"] == 1 and row["day_of_month"] in [24, 25, 26]:
                if "Salary" in categories:
                    idx = list(categories).index("Salary")
                    y_pred_proba[i] = 0
                    y_pred_proba[i][idx] = 1
            
            # Rule 2: Rent - Withdrawal 3900–7500 и дата ≤6
            elif row["is_rent_like"] == 1:
                if "Rent" in categories:
                    idx = list(categories).index("Rent")
                    y_pred_proba[i] = 0
                    y_pred_proba[i][idx] = 1
            
            # Rule 3: Shopping - 150–3000 в первые 5 дней после зарплаты
            elif (
                row["Withdrawal"] >= 150 and
                row["Withdrawal"] <= 3000 and
                row["days_since_last_salary"] >= 0 and
                row["days_since_last_salary"] <= 5 and
                row["is_withdrawal"] == 1
            ):
                if "Shopping" in categories:
                    idx = list(categories).index("Shopping")
                    y_pred_proba[i] = 0
                    y_pred_proba[i][idx] = 1
        
        # Получаем предсказания и вероятности
        pred_indices = np.argmax(y_pred_proba, axis=1)
        pred_categories = categories[pred_indices]
        max_probas = np.max(y_pred_proba, axis=1)
        
        return pred_categories, max_probas
    
    def _map_to_transaction_category(self, prediction: str) -> TransactionCategory:
        """Преобразование предсказания модели в TransactionCategory"""
        return CATEGORY_MAPPING.get(prediction, TransactionCategory.OTHER_EXPENSE)
    
    def train(self, csv_path: str, **kwargs) -> Dict[str, Any]:
        """
        Обучение модели на данных из CSV
        
        Args:
            csv_path: Путь к CSV файлу с данными
            **kwargs: Дополнительные параметры для обучения
            
        Returns:
            Словарь с метриками обучения
        """
        # Загрузка данных
        df = pd.read_csv(csv_path, skiprows=5)
        df.columns = ["Date1", "Category", "RefNo", "Date2", "Withdrawal", "Deposit", "Balance"]
        df["Date"] = pd.to_datetime(df["Date2"], format="mixed", errors="coerce")
        df = df[["Date", "Category", "Withdrawal", "Deposit", "Balance"]].copy()
        df["Withdrawal"] = pd.to_numeric(df["Withdrawal"], errors="coerce").fillna(0)
        df["Deposit"] = pd.to_numeric(df["Deposit"], errors="coerce").fillna(0)
        df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce").fillna(method="ffill")
        
        print(f"📊 Загружено {len(df)} записей")
        
        # Фильтрация категорий
        df["Category"] = df["Category"].replace("Transport", "Misc")
        df = df[df["Category"].isin(VALID_CATEGORIES)].copy().reset_index(drop=True)
        
        print(f"📊 После фильтрации: {len(df)} записей")
        
        # Извлечение признаков
        df_features = self._extract_features(df)
        
        # Подготовка X и y
        feature_columns = self._get_feature_columns()
        X = df_features[feature_columns].fillna(-1)
        y = df_features["Category"]
        
        # Разделение по времени: train до июля, test — июль–декабрь
        split_date = "2023-07-01"
        train_idx = df_features[df_features["Date"] < split_date].index
        test_idx = df_features[df_features["Date"] >= split_date].index
        
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
        
        print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Обучение модели
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            num_leaves=15,
            learning_rate=0.05,
            min_data_in_leaf=10,
            lambda_l1=0.1,
            lambda_l2=0.1,
            random_state=42,
            class_weight="balanced"
        )
        
        print("🚀 Обучение модели...")
        self.model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred_hybrid, _ = self._apply_fallback_rules(X_test, y_pred_proba)
        
        # Метрики
        accuracy = (y_pred_hybrid == y_test).mean()
        f1_weighted = f1_score(y_test, y_pred_hybrid, average="weighted")
        
        print(f"✅ Точность (Accuracy): {accuracy:.4f}")
        print(f"✅ F1-score (weighted): {f1_weighted:.4f}")
        
        self.is_trained = True
        self.save_model()
        
        return {
            'accuracy': float(accuracy),
            'f1_weighted': float(f1_weighted),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'is_trained': True
        }
    
    def predict(self, description: str, amount: float, is_expense: bool = True, 
                date: Optional[datetime] = None) -> Tuple[TransactionCategory, float]:
        """
        Предсказание категории для одной транзакции
        
        Args:
            description: Описание транзакции (RefNo) - не используется в этой модели
            amount: Сумма транзакции
            is_expense: Является ли транзакция расходом
            date: Дата транзакции
            
        Returns:
            Tuple с категорией и вероятностью
        """
        if not self.is_trained or self.model is None:
            return TransactionCategory.OTHER_EXPENSE, 0.5
        
        # Создаем DataFrame для одной транзакции
        if date is None:
            date = datetime.now()
        
        withdrawal = amount if is_expense else 0.0
        deposit = amount if not is_expense else 0.0
        
        # Для одной транзакции баланс неизвестен, используем 0
        df = pd.DataFrame([{
            'Date': date,
            'Withdrawal': withdrawal,
            'Deposit': deposit,
            'Balance': 0.0  # Неизвестен для одной транзакции
        }])
        
        # Извлекаем признаки
        df_features = self._extract_features(df)
        
        # Подготовка признаков
        feature_columns = self._get_feature_columns()
        X = df_features[feature_columns].fillna(-1)
        
        # Предсказание
        y_pred_proba = self.model.predict_proba(X)
        y_pred_hybrid, y_proba_hybrid = self._apply_fallback_rules(X, y_pred_proba)
        
        # Преобразование в TransactionCategory
        category = self._map_to_transaction_category(y_pred_hybrid[0])
        probability = float(y_proba_hybrid[0])
        
        return category, probability
    
    def predict_from_dataframe(self, df: pd.DataFrame) -> Tuple[TransactionCategory, float]:
        """
        Предсказание категории из DataFrame (для API endpoint)
        
        Args:
            df: DataFrame с колонками Date, RefNo, Withdrawal, Deposit, Balance
            
        Returns:
            Tuple с категорией и вероятностью
        """
        if not self.is_trained or self.model is None:
            return TransactionCategory.OTHER_EXPENSE, 0.5
        
        # Обработка DataFrame
        if 'Date' not in df.columns:
            df['Date'] = pd.to_datetime(df.get('Date', datetime.now()), errors='coerce', dayfirst=True, format='mixed')
            df['Date'] = df['Date'].fillna(pd.Timestamp.now())
        else:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True, format='mixed')
            df['Date'] = df['Date'].fillna(pd.Timestamp.now())
        
        # Убеждаемся, что числовые колонки корректны
        df['Withdrawal'] = pd.to_numeric(df['Withdrawal'], errors='coerce').fillna(0)
        df['Deposit'] = pd.to_numeric(df['Deposit'], errors='coerce').fillna(0)
        df['Balance'] = pd.to_numeric(df.get('Balance', 0), errors='coerce').fillna(0)
        
        # Берем первую строку для предсказания
        if len(df) == 0:
            return TransactionCategory.OTHER_EXPENSE, 0.5
        
        df_single = df.iloc[[0]].copy()
        
        # Извлекаем признаки
        df_features = self._extract_features(df_single)
        
        # Подготовка признаков
        feature_columns = self._get_feature_columns()
        X = df_features[feature_columns].fillna(-1)
        
        # Предсказание
        y_pred_proba = self.model.predict_proba(X)
        y_pred_hybrid, y_proba_hybrid = self._apply_fallback_rules(X, y_pred_proba)
        
        # Преобразование в TransactionCategory
        category = self._map_to_transaction_category(y_pred_hybrid[0])
        probability = float(y_proba_hybrid[0])
        
        return category, probability
    
    def save_model(self):
        """Сохранение модели в файл"""
        if self.model is None:
            print("⚠️ Модель не обучена, нечего сохранять")
            return
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"✅ Модель сохранена в: {self.model_path}")
    
    def load_model(self):
        """Загрузка модели из файла"""
        if not os.path.exists(self.model_path):
            print(f"❌ Файл модели не существует: {self.model_path}")
            return
        
        try:
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            print(f"✅ Модель загружена из: {self.model_path}")
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
            import traceback
            traceback.print_exc()
            self.is_trained = False
