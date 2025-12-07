"""Классификатор транзакций - использует логику из test_classifier.py"""
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

from app.models.category import TransactionCategory
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score

# Маппинг категорий модели в категории системы
CATEGORY_MAPPING = {
    'Rent': TransactionCategory.UTILITIES,
    'Misc': TransactionCategory.OTHER_EXPENSE,
    'Food': TransactionCategory.PRODUCTS,
    'Salary': TransactionCategory.SALARY,
    'Shopping': TransactionCategory.CLOTHING,
}


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
            # Путь по умолчанию - модель в папке ml
            default_path = Path(__file__).parent / "classifier_v2.pkl"
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
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение признаков (логика из test_classifier.py)
        
        Args:
            df: DataFrame с колонками Date, Withdrawal, Deposit, Balance
            
        Returns:
            DataFrame с извлеченными признаками
        """
        # Убеждаемся, что Date является datetime типом
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
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

        # Точные бизнес-правила
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
    
    def apply_enhanced_fallback_rules_with_proba(self, X: pd.DataFrame, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Применение fallback правил (логика из test_classifier.py)
        
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
            # Salary: Deposit == 34800 и дата 24–26
            if row["is_salary_like"] == 1 and row["day_of_month"] in [24, 25, 26]:
                idx = list(categories).index("Salary")
                y_pred_proba[i] = 0
                y_pred_proba[i][idx] = 1
            # Rent: Withdrawal 3900–7500 и дата ≤6
            elif row["is_rent_like"] == 1:
                idx = list(categories).index("Rent")
                y_pred_proba[i] = 0
                y_pred_proba[i][idx] = 1
            # Shopping: 150–3000 в первые 5 дней после зарплаты
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
        
        pred_indices = np.argmax(y_pred_proba, axis=1)
        pred_categories = categories[pred_indices]
        max_probas = np.max(y_pred_proba, axis=1)
        return pred_categories, max_probas
    
    def predict_from_dataframe(self, df: pd.DataFrame) -> Tuple[TransactionCategory, float]:
        """
        Предсказание категории из DataFrame (логика из test_classifier.py)
        
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
        df_features = self.extract_features(df_single)
        
        # Подготовка признаков
        feature_columns = [
            "is_withdrawal", "is_deposit", "amount", "net_flow", "balance_before",
            "day_of_month", "day_of_week", "is_month_start", "is_month_end",
            "is_salary_like", "is_rent_like", "days_since_last_salary", "days_since_last_txn",
            "Withdrawal", "Deposit"
        ]
        
        X = df_features[feature_columns]
        X = X.fillna(-1)  # Заполняем NaN
        
        # Предсказание
        y_pred_proba = self.model.predict_proba(X)
        y_pred_hybrid, y_proba_hybrid = self.apply_enhanced_fallback_rules_with_proba(X, y_pred_proba)
        
        # Преобразование в TransactionCategory
        category_str = y_pred_hybrid[0]
        category = CATEGORY_MAPPING.get(category_str, TransactionCategory.OTHER_EXPENSE)
        probability = float(y_proba_hybrid[0])
        
        return category, probability
    
    def train(self, df: pd.DataFrame, original_csv_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Обучение модели на DataFrame (логика из cybergarden_ML.py)
        
        Args:
            df: DataFrame с колонками Date, Category, Withdrawal, Deposit, Balance
            original_csv_path: Путь к оригинальному CSV файлу (опционально, для объединения данных)
            
        Returns:
            Словарь с метриками обучения
        """
        print("🚀 Начало обучения модели...")
        
        # Если указан оригинальный CSV, объединяем данные
        if original_csv_path and os.path.exists(original_csv_path):
            print(f"📂 Загрузка оригинальных данных из {original_csv_path}...")
            try:
                df_original = pd.read_csv(original_csv_path, skiprows=5)
                df_original.columns = ["Date1", "Category", "RefNo", "Date2", "Withdrawal", "Deposit", "Balance"]
                df_original["Date"] = pd.to_datetime(df_original["Date2"], format="mixed", errors="coerce")
                df_original = df_original[["Date", "Category", "Withdrawal", "Deposit", "Balance"]].copy()
                df_original["Withdrawal"] = pd.to_numeric(df_original["Withdrawal"], errors="coerce").fillna(0)
                df_original["Deposit"] = pd.to_numeric(df_original["Deposit"], errors="coerce").fillna(0)
                df_original["Balance"] = pd.to_numeric(df_original["Balance"], errors="coerce").fillna(method="ffill")
                
                # Объединяем данные
                df = pd.concat([df_original, df], ignore_index=True)
                print(f"✅ Объединено данных: {len(df_original)} (оригинал) + {len(df) - len(df_original)} (новые) = {len(df)}")
            except Exception as e:
                print(f"⚠️ Не удалось загрузить оригинальный CSV: {e}. Используем только новые данные.")
        
        # Убеждаемся, что Date является datetime типом (если еще не преобразовано)
        if 'Date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Очистка данных (логика из cybergarden_ML.py)
        df["Category"] = df["Category"].replace("Transport", "Misc")
        valid_cats = ["Food", "Misc", "Rent", "Salary", "Shopping"]
        df = df[df["Category"].isin(valid_cats)].copy().reset_index(drop=True)
        
        if len(df) == 0:
            raise ValueError("Нет данных для обучения после фильтрации категорий")
        
        print(f"📊 Данных для обучения: {len(df)}")
        print(f"📊 Распределение категорий:\n{df['Category'].value_counts()}")
        
        # Извлечение признаков
        df_features = self.extract_features(df)
        
        # Подготовка X и y
        feature_columns = [
            "is_withdrawal", "is_deposit", "amount", "net_flow", "balance_before",
            "day_of_month", "day_of_week", "is_month_start", "is_month_end",
            "is_salary_like", "is_rent_like", "days_since_last_salary", "days_since_last_txn",
            "Withdrawal", "Deposit"
        ]
        
        X = df_features[feature_columns].fillna(-1)
        y = df_features["Category"]
        
        # Разделение по времени: train до 70% данных, test — остальные 30%
        split_idx = int(len(df_features) * 0.7)
        train_idx = df_features.index[:split_idx]
        test_idx = df_features.index[split_idx:]
        
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
        
        print(f"📊 Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Обучение модели (логика из cybergarden_ML.py)
        model = lgb.LGBMClassifier(
            n_estimators=100,
            num_leaves=15,
            learning_rate=0.05,
            min_data_in_leaf=10,
            lambda_l1=0.1,
            lambda_l2=0.1,
            random_state=42,
            class_weight="balanced"
        )
        
        print("🎯 Обучение модели...")
        model.fit(X_train, y_train)
        
        # Применение fallback правил
        def apply_fallback_rules(X, y_pred, categories):
            y_pred = y_pred.copy()
            for i in range(len(X)):
                row = X.iloc[i]
                if row["is_salary_like"] == 1 and row["day_of_month"] in [24, 25, 26]:
                    y_pred[i] = "Salary"
                elif row["is_rent_like"] == 1:
                    y_pred[i] = "Rent"
                elif (
                    row["Withdrawal"] >= 150 and
                    row["Withdrawal"] <= 3000 and
                    row["days_since_last_salary"] >= 0 and
                    row["days_since_last_salary"] <= 5 and
                    row["is_withdrawal"] == 1
                ):
                    y_pred[i] = "Shopping"
            return y_pred
        
        y_pred = model.predict(X_test)
        y_pred_hybrid = apply_fallback_rules(X_test, y_pred, model.classes_)
        
        # Оценка модели
        f1_weighted = f1_score(y_test, y_pred_hybrid, average="weighted", zero_division=0)
        f1_macro = f1_score(y_test, y_pred_hybrid, average="macro", zero_division=0)
        
        print("\n=== Метрики модели ===")
        print(classification_report(y_test, y_pred_hybrid, zero_division=0))
        print(f"F1 (weighted): {f1_weighted:.4f}")
        print(f"F1 (macro): {f1_macro:.4f}")
        
        # Сохранение модели
        self.model = model
        self.save_model()
        self.is_trained = True
        
        print(f"✅ Модель обучена и сохранена в: {self.model_path}")
        
        return {
            "f1_weighted": float(f1_weighted),
            "f1_macro": float(f1_macro),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "total_samples": len(df),
            "categories": valid_cats
        }
    
    def save_model(self):
        """Сохранение модели в файл"""
        if self.model is None:
            print("❌ Нет модели для сохранения")
            return
        
        try:
            # Создаем директорию, если её нет
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            print(f"✅ Модель сохранена в: {self.model_path}")
        except Exception as e:
            print(f"❌ Ошибка при сохранении модели: {e}")
            import traceback
            traceback.print_exc()
    
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
