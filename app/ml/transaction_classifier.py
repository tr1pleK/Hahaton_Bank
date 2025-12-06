"""Модель машинного обучения для классификации банковских транзакций"""
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import hstack

from app.models.category import TransactionCategory


# Маппинг категорий из CSV в категории системы
CATEGORY_MAPPING = {
    'Rent': TransactionCategory.UTILITIES,
    'Misc': TransactionCategory.OTHER_EXPENSE,
    'Food': TransactionCategory.PRODUCTS,
    'Salary': TransactionCategory.SALARY,
    'Shopping': TransactionCategory.CLOTHING,
    'Transport': TransactionCategory.TRANSPORT,
}


class TransactionClassifier:
    """Классификатор транзакций на основе машинного обучения"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация классификатора
        
        Args:
            model_path: Путь к сохраненной модели (опционально)
        """
        self.model_path = model_path or self._get_default_model_path()
        self.model: Optional[Pipeline] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_trained = False
        
        # Загрузить модель, если она существует
        if os.path.exists(self.model_path):
            print(f"🔍 Найдена модель по пути: {self.model_path}")
            self.load_model()
            if self.is_trained:
                print(f"✅ Модель успешно загружена и готова к использованию")
            else:
                print(f"⚠️ Модель найдена, но не помечена как обученная")
        else:
            print(f"⚠️ Модель не найдена по пути: {self.model_path}")
            print(f"   Используется fallback классификация (вероятность всегда 0.5)")
    
    @staticmethod
    def _get_default_model_path() -> str:
        """Получить путь по умолчанию для сохранения модели"""
        # Сначала проверяем относительный путь (работает в Docker и на хосте)
        base_dir = Path(__file__).parent.parent.parent
        models_dir = base_dir / "ml_models"
        models_dir.mkdir(exist_ok=True)
        relative_path = str(models_dir / "transaction_classifier.pkl")
        
        # Если файл существует по относительному пути, используем его
        if os.path.exists(relative_path):
            return relative_path
        
        # Для Windows хоста проверяем абсолютный путь
        if os.name == 'nt':  # Windows
            windows_path = r"C:\Users\Егор\IdeaProjects\Hahaton_Bank\backend\ml_models\transaction_classifier.pkl"
            if os.path.exists(windows_path):
                return windows_path
        
        # Возвращаем относительный путь (будет использован для сохранения новой модели)
        return relative_path
    
    def _preprocess_text(self, text: str) -> str:
        """Предобработка текстового описания"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower()
        # Удаление специальных символов, оставляем только буквы, цифры и пробелы
        text = re.sub(r'[^a-zа-яё0-9\s]', ' ', text)
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Извлечение признаков из данных
        
        Returns:
            Tuple с признаками и метками
        """
        # Обработка текстового описания (RefNo)
        df['description_clean'] = df['RefNo'].apply(self._preprocess_text)
        
        # Определение суммы транзакции (Withdrawal или Deposit)
        def get_amount(row):
            """Безопасное извлечение суммы из строки"""
            try:
                withdrawal = row.get('Withdrawal', 0)
                deposit = row.get('Deposit', 0)
                
                # Преобразование в число, если это строка
                if pd.notna(withdrawal):
                    withdrawal = float(withdrawal) if withdrawal != '' else 0.0
                else:
                    withdrawal = 0.0
                    
                if pd.notna(deposit):
                    deposit = float(deposit) if deposit != '' else 0.0
                else:
                    deposit = 0.0
                
                # Возвращаем сумму (положительное значение)
                if withdrawal > 0:
                    return withdrawal
                elif deposit > 0:
                    return deposit
                else:
                    return 0.0
            except (ValueError, TypeError):
                return 0.0
        
        df['amount'] = df.apply(get_amount, axis=1)
        
        # Определение типа транзакции (расход или доход)
        def get_is_expense(row):
            """Определение, является ли транзакция расходом"""
            try:
                withdrawal = row.get('Withdrawal', 0)
                if pd.notna(withdrawal):
                    withdrawal = float(withdrawal) if withdrawal != '' else 0.0
                    return 1 if withdrawal > 0 else 0
                return 0
            except (ValueError, TypeError):
                return 0
        
        df['is_expense'] = df.apply(get_is_expense, axis=1)
        
        # Извлечение признаков из даты
        df['date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True, format='mixed')
        # Заполнение NaN значений даты текущей датой
        df['date'] = df['date'].fillna(pd.Timestamp.now())
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        
        # Маппинг категорий
        df['category_mapped'] = df['Category'].map(CATEGORY_MAPPING)
        df = df.dropna(subset=['category_mapped'])
        
        # Удаление строк с NaN в числовых признаках
        df = df.dropna(subset=['amount', 'day_of_week', 'day_of_month', 'month'])
        
        if len(df) == 0:
            raise ValueError("Нет данных после обработки признаков")
        
        # Признаки для модели
        X_text = df['description_clean'].values
        X_numeric = df[['amount', 'is_expense', 'day_of_week', 'day_of_month', 'month']].values
        y = df['category_mapped'].apply(lambda x: x.value).values
        
        return (X_text, X_numeric), y
    
    def _create_model(self) -> Pipeline:
        """Создание модели машинного обучения"""
        # TF-IDF для текстовых признаков
        text_transformer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=2,
            stop_words=None
        )
        
        # Стандартизация для числовых признаков
        numeric_transformer = StandardScaler()
        
        # Комбинированный трансформер
        preprocessor = ColumnTransformer(
            transformers=[
                ('text', text_transformer, 0),
                ('numeric', numeric_transformer, 1)
            ],
            remainder='passthrough'
        )
        
        # Классификатор
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Создаем пайплайн
        # Примечание: ColumnTransformer требует особой обработки для текстовых и числовых данных
        # Поэтому создадим упрощенную версию
        return {
            'text_transformer': text_transformer,
            'numeric_transformer': numeric_transformer,
            'classifier': classifier
        }
    
    def train(self, csv_path: str, test_size: float = 0.2, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Обучение модели на данных из CSV
        
        Args:
            csv_path: Путь к CSV файлу с данными
            test_size: Доля тестовой выборки
            force_retrain: Если True, переобучает модель даже если файл существует
            
        Returns:
            Словарь с метриками обучения
        """
        # Проверка существования модели
        if not force_retrain and os.path.exists(self.model_path):
            print(f"📦 Модель уже существует по пути: {self.model_path}")
            print(f"   Загружаем существующую модель вместо обучения новой...")
            
            # Загружаем существующую модель
            self.load_model()
            
            if self.is_trained and self.model is not None:
                print(f"✅ Существующая модель успешно загружена")
                print(f"   Для переобучения используйте параметр force_retrain=True")
                
                # Возвращаем информацию о загруженной модели
                return {
                    'message': 'Модель уже обучена и загружена из файла',
                    'model_path': self.model_path,
                    'is_trained': True,
                    'loaded_from_file': True
                }
            else:
                print(f"⚠️ Файл модели существует, но модель не может быть загружена")
                print(f"   Продолжаем обучение новой модели...")
        
        # Загрузка данных
        df = pd.read_csv(csv_path)
        
        # Удаление полностью пустых строк
        df = df.dropna(how='all')
        
        # Удаление строк с пустыми категориями или RefNo
        df = df.dropna(subset=['Category', 'RefNo'])
        
        # Фильтрация только известных категорий
        df = df[df['Category'].isin(CATEGORY_MAPPING.keys())]
        
        # Удаление строк, где нет ни Withdrawal, ни Deposit
        df = df[
            (pd.notna(df['Withdrawal']) & (df['Withdrawal'] != 0)) | 
            (pd.notna(df['Deposit']) & (df['Deposit'] != 0))
        ]
        
        if len(df) == 0:
            raise ValueError("Нет данных для обучения после фильтрации")
        
        # Извлечение признаков
        (X_text, X_numeric), y = self._extract_features(df)
        
        # Разделение на обучающую и тестовую выборки
        X_text_train, X_text_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
            X_text, X_numeric, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Создание и обучение трансформеров
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=2,
            stop_words=None
        )
        X_text_train_tfidf = self.vectorizer.fit_transform(X_text_train)
        X_text_test_tfidf = self.vectorizer.transform(X_text_test)
        
        self.scaler = StandardScaler()
        X_numeric_train_scaled = self.scaler.fit_transform(X_numeric_train)
        X_numeric_test_scaled = self.scaler.transform(X_numeric_test)
        
        # Объединение признаков
        X_train = hstack([X_text_train_tfidf, X_numeric_train_scaled])
        X_test = hstack([X_text_test_tfidf, X_numeric_test_scaled])
        
        # Обучение классификатора
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        self.is_trained = True
        
        # Сохранение модели
        self.save_model()
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'train_samples': len(y_train),
            'test_samples': len(y_test)
        }
    
    def predict(self, description: str, amount: float, is_expense: bool = True, 
                date: Optional[datetime] = None) -> TransactionCategory:
        """
        Предсказание категории транзакции
        
        Args:
            description: Описание транзакции (RefNo или описание)
            amount: Сумма транзакции
            is_expense: Является ли транзакция расходом
            date: Дата транзакции (опционально)
            
        Returns:
            Предсказанная категория
        """
        if not self.is_trained or self.model is None:
            # Если модель не обучена, используем простую эвристику
            return self._fallback_classify(description, amount, is_expense)
        
        # Предобработка текста
        text_clean = self._preprocess_text(description)
        
        # Извлечение признаков из даты
        if date is None:
            date = datetime.now()
        
        day_of_week = date.weekday()
        day_of_month = date.day
        month = date.month
        
        # Преобразование текста
        text_tfidf = self.vectorizer.transform([text_clean])
        
        # Преобразование числовых признаков
        numeric_features = np.array([[amount, 1 if is_expense else 0, day_of_week, day_of_month, month]])
        numeric_scaled = self.scaler.transform(numeric_features)
        
        # Объединение признаков
        X = hstack([text_tfidf, numeric_scaled])
        
        # Предсказание
        prediction = self.model.predict(X)[0]
        
        # Преобразование обратно в TransactionCategory
        for cat in TransactionCategory:
            if cat.value == prediction:
                return cat
        
        return TransactionCategory.OTHER_EXPENSE
    
    def predict_with_probability(
        self, 
        description: str, 
        amount: float, 
        is_expense: bool = True, 
        date: Optional[datetime] = None
    ) -> Tuple[TransactionCategory, float]:
        """
        Предсказание категории транзакции с вероятностью
        
        Args:
            description: Описание транзакции (RefNo или описание)
            amount: Сумма транзакции
            is_expense: Является ли транзакция расходом
            date: Дата транзакции (опционально)
            
        Returns:
            Tuple с предсказанной категорией и вероятностью
        """
        if not self.is_trained or self.model is None:
            # Если модель не обучена, используем простую эвристику
            category = self._fallback_classify(description, amount, is_expense)
            return category, 0.5  # Низкая уверенность для fallback
        
        # Предобработка текста
        text_clean = self._preprocess_text(description)
        
        # Извлечение признаков из даты
        if date is None:
            date = datetime.now()
        
        day_of_week = date.weekday()
        day_of_month = date.day
        month = date.month
        
        # Преобразование текста
        text_tfidf = self.vectorizer.transform([text_clean])
        
        # Преобразование числовых признаков
        numeric_features = np.array([[amount, 1 if is_expense else 0, day_of_week, day_of_month, month]])
        numeric_scaled = self.scaler.transform(numeric_features)
        
        # Объединение признаков
        X = hstack([text_tfidf, numeric_scaled])
        
        # Предсказание с вероятностями
        probabilities = self.model.predict_proba(X)
        prediction = self.model.predict(X)[0]
        
        # probabilities может быть 2D массивом (n_samples, n_classes)
        # Берем первую строку (первый образец)
        if probabilities.ndim > 1:
            prob_array = probabilities[0]
        else:
            prob_array = probabilities
        
        # Находим индекс предсказанного класса в model.classes_
        # model.classes_ содержит все классы в том же порядке, что и вероятности
        prediction_idx = None
        if hasattr(self.model, 'classes_'):
            try:
                prediction_idx = np.where(self.model.classes_ == prediction)[0]
                if len(prediction_idx) > 0:
                    prediction_idx = prediction_idx[0]
                else:
                    prediction_idx = None
            except Exception:
                prediction_idx = None
        
        # Извлекаем вероятность для предсказанного класса
        if prediction_idx is not None and prediction_idx < len(prob_array):
            max_probability = float(prob_array[prediction_idx])
        else:
            # Если не нашли индекс, берем максимальную вероятность
            max_probability = float(np.max(prob_array))
        
        # Убеждаемся, что вероятность в диапазоне [0, 1]
        # predict_proba всегда возвращает нормализованные вероятности, но на всякий случай
        max_probability = max(0.0, min(1.0, max_probability))
        
        # Преобразование обратно в TransactionCategory
        for cat in TransactionCategory:
            if cat.value == prediction:
                return cat, max_probability
        
        return TransactionCategory.OTHER_EXPENSE, max_probability
    
    def predict_from_dataframe(self, df: pd.DataFrame) -> Tuple[TransactionCategory, float]:
        """
        Предсказание категории из DataFrame (как при обучении)
        
        Args:
            df: DataFrame с колонками Date, RefNo, Withdrawal, Deposit, Balance
            
        Returns:
            Tuple с предсказанной категорией и вероятностью
        """
        # Проверка состояния модели
        if not self.is_trained:
            print(f"⚠️ Модель не обучена (is_trained=False), используется fallback")
        if self.model is None:
            print(f"⚠️ Модель не загружена (model=None), используется fallback")
        if self.vectorizer is None:
            print(f"⚠️ Vectorizer не загружен, используется fallback")
        if self.scaler is None:
            print(f"⚠️ Scaler не загружен, используется fallback")
        
        if not self.is_trained or self.model is None or self.vectorizer is None or self.scaler is None:
            # Если модель не обучена, используем простую эвристику
            if len(df) == 0:
                return TransactionCategory.OTHER_EXPENSE, 0.5
            
            row = df.iloc[0]
            description = str(row.get('RefNo', ''))
            amount = float(row.get('Withdrawal', 0) or row.get('Deposit', 0) or 0)
            is_expense = float(row.get('Withdrawal', 0) or 0) > 0
            category = self._fallback_classify(description, amount, is_expense)
            print(f"🔄 Использована fallback классификация: {category.value} (вероятность: 0.5)")
            return category, 0.5
        
        # Применяем ту же логику обработки признаков, что при обучении
        # Обработка текстового описания (RefNo)
        df['description_clean'] = df['RefNo'].apply(self._preprocess_text)
        
        # Определение суммы транзакции (Withdrawal или Deposit)
        def get_amount(row):
            """Безопасное извлечение суммы из строки"""
            try:
                withdrawal = row.get('Withdrawal', 0)
                deposit = row.get('Deposit', 0)
                
                # Преобразование в число, если это строка
                if pd.notna(withdrawal):
                    withdrawal = float(withdrawal) if withdrawal != '' else 0.0
                else:
                    withdrawal = 0.0
                    
                if pd.notna(deposit):
                    deposit = float(deposit) if deposit != '' else 0.0
                else:
                    deposit = 0.0
                
                # Возвращаем сумму (положительное значение)
                if withdrawal > 0:
                    return withdrawal
                elif deposit > 0:
                    return deposit
                else:
                    return 0.0
            except (ValueError, TypeError):
                return 0.0
        
        df['amount'] = df.apply(get_amount, axis=1)
        
        # Определение типа транзакции (расход или доход)
        def get_is_expense(row):
            """Определение, является ли транзакция расходом"""
            try:
                withdrawal = row.get('Withdrawal', 0)
                if pd.notna(withdrawal):
                    withdrawal = float(withdrawal) if withdrawal != '' else 0.0
                    return 1 if withdrawal > 0 else 0
                return 0
            except (ValueError, TypeError):
                return 0
        
        df['is_expense'] = df.apply(get_is_expense, axis=1)
        
        # Извлечение признаков из даты
        df['date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True, format='mixed')
        # Заполнение NaN значений даты текущей датой
        df['date'] = df['date'].fillna(pd.Timestamp.now())
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        
        # Удаление строк с NaN в числовых признаках
        df = df.dropna(subset=['amount', 'day_of_week', 'day_of_month', 'month'])
        
        if len(df) == 0:
            return TransactionCategory.OTHER_EXPENSE, 0.5
        
        # Берем первую строку для предсказания
        row = df.iloc[0]
        
        # Проверяем тип vectorizer
        from sklearn.preprocessing import LabelEncoder
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        text_clean = row['description_clean']
        refno_original = str(row.get('RefNo', ''))
        
        # Если vectorizer это LabelEncoder, значит модель была обучена по-другому
        # LabelEncoder кодирует RefNo как категориальную переменную
        if isinstance(self.vectorizer, LabelEncoder):
            try:
                # Пробуем закодировать RefNo через LabelEncoder
                # Если RefNo не найден, используем значение по умолчанию (0 или -1)
                if hasattr(self.vectorizer, 'classes_'):
                    # Проверяем, есть ли RefNo в обученных классах
                    if refno_original in self.vectorizer.classes_:
                        refno_encoded = self.vectorizer.transform([refno_original])[0]
                    else:
                        # Если RefNo не найден, используем значение по умолчанию
                        # Используем максимальное значение + 1 или среднее значение
                        # Для простоты используем -1 как маркер неизвестного значения
                        # Модель должна была быть обучена с учетом таких случаев
                        if len(self.vectorizer.classes_) > 0:
                            # Используем максимальное закодированное значение + 1
                            # или можно использовать -1 как маркер неизвестного
                            max_encoded = len(self.vectorizer.classes_) - 1
                            # Используем значение немного больше максимального
                            refno_encoded = max_encoded + 1
                        else:
                            refno_encoded = 0
                else:
                    # Если нет classes_, пробуем transform напрямую
                    try:
                        refno_encoded = self.vectorizer.transform([refno_original])[0]
                    except ValueError:
                        # Если transform не работает, используем 0
                        refno_encoded = 0
                
                # LabelEncoder возвращает одно число, нужно преобразовать в массив признаков
                # В зависимости от того, как модель была обучена, это может быть просто число
                # или нужно создать one-hot encoding
                # Для простоты, используем закодированное значение как признак
                refno_feature = np.array([[float(refno_encoded)]])
                
            except Exception as e:
                print(f"⚠️ Ошибка при кодировании RefNo через LabelEncoder: {e}")
                print(f"   Используется fallback классификация")
                description = str(row.get('RefNo', ''))
                amount = float(row.get('amount', 0))
                is_expense = bool(row.get('is_expense', 0))
                category = self._fallback_classify(description, amount, is_expense)
                return category, 0.5
        
        # Если vectorizer это TfidfVectorizer, используем стандартный подход
        elif isinstance(self.vectorizer, TfidfVectorizer):
            try:
                text_tfidf = self.vectorizer.transform([text_clean])
            except Exception as e:
                print(f"⚠️ Ошибка при преобразовании текста: {e}")
                description = str(row.get('RefNo', ''))
                amount = float(row.get('amount', 0))
                is_expense = bool(row.get('is_expense', 0))
                category = self._fallback_classify(description, amount, is_expense)
                return category, 0.5
        else:
            # Неизвестный тип vectorizer
            print(f"⚠️ Неожиданный тип vectorizer: {type(self.vectorizer)}")
            print(f"   Используется fallback классификация")
            description = str(row.get('RefNo', ''))
            amount = float(row.get('amount', 0))
            is_expense = bool(row.get('is_expense', 0))
            category = self._fallback_classify(description, amount, is_expense)
            return category, 0.5
        
        # Преобразование числовых признаков
        numeric_features = np.array([[ 
            row['amount'],
            row['is_expense'],
            row['day_of_week'],
            row['day_of_month'],
            row['month']
        ]])
        
        # Проверяем тип scaler
        from sklearn.preprocessing import StandardScaler
        if isinstance(self.scaler, StandardScaler):
            try:
                numeric_scaled = self.scaler.transform(numeric_features)
            except Exception as e:
                print(f"⚠️ Ошибка при масштабировании признаков: {e}")
                description = str(row.get('RefNo', ''))
                amount = float(row.get('amount', 0))
                is_expense = bool(row.get('is_expense', 0))
                category = self._fallback_classify(description, amount, is_expense)
                return category, 0.5
        elif isinstance(self.scaler, list):
            # Если scaler это list, возможно это список параметров для масштабирования
            # Используем признаки без масштабирования или применяем простую нормализацию
            print(f"   Scaler имеет тип list, используем признаки без масштабирования")
            numeric_scaled = numeric_features
        else:
            # Если scaler это не StandardScaler, используем признаки без масштабирования
            print(f"   Scaler имеет неожиданный тип: {type(self.scaler)}, используем признаки без масштабирования")
            numeric_scaled = numeric_features
        
        # Объединение признаков в зависимости от типа vectorizer
        if isinstance(self.vectorizer, LabelEncoder):
            # Для LabelEncoder объединяем закодированный RefNo с числовыми признаками
            # refno_feature уже является массивом формы (1, 1)
            # numeric_scaled имеет форму (1, 5)
            # Объединяем их по горизонтали
            X = np.hstack([refno_feature, numeric_scaled])
        elif isinstance(self.vectorizer, TfidfVectorizer):
            # Для TfidfVectorizer объединяем TF-IDF векторы с числовыми признаками
            X = hstack([text_tfidf, numeric_scaled])
        else:
            # Если неизвестный тип, используем только числовые признаки
            print(f"⚠️ Неизвестный тип vectorizer, используем только числовые признаки")
            X = numeric_scaled
        
        # Предсказание с вероятностями
        try:
            probabilities = self.model.predict_proba(X)
            prediction = self.model.predict(X)[0]
            
            # probabilities может быть 2D массивом (n_samples, n_classes)
            # Берем первую строку (первый образец)
            if probabilities.ndim > 1:
                prob_array = probabilities[0]
            else:
                prob_array = probabilities
            
            # Находим индекс предсказанного класса в model.classes_
            # model.classes_ содержит все классы в том же порядке, что и вероятности
            prediction_idx = None
            if hasattr(self.model, 'classes_'):
                try:
                    prediction_idx = np.where(self.model.classes_ == prediction)[0]
                    if len(prediction_idx) > 0:
                        prediction_idx = prediction_idx[0]
                    else:
                        prediction_idx = None
                except Exception as e:
                    print(f"   ⚠️ Ошибка при поиске индекса класса: {e}")
                    prediction_idx = None
            
            # Извлекаем вероятность для предсказанного класса
            if prediction_idx is not None and prediction_idx < len(prob_array):
                max_probability = float(prob_array[prediction_idx])
            else:
                # Если не нашли индекс, берем максимальную вероятность
                max_probability = float(np.max(prob_array))
            
            # Убеждаемся, что вероятность в диапазоне [0, 1]
            # predict_proba всегда возвращает нормализованные вероятности, но на всякий случай
            max_probability = max(0.0, min(1.0, max_probability))
            
            # Логирование для отладки (можно убрать после проверки)
            if hasattr(self.model, 'classes_'):
                print(f"   Предсказание: {prediction}, индекс: {prediction_idx}, вероятность: {max_probability:.4f}")
                print(f"   Доступные классы: {self.model.classes_}")
                print(f"   Все вероятности: {prob_array}")
            
            # Преобразование обратно в TransactionCategory
            for cat in TransactionCategory:
                if cat.value == prediction:
                    return cat, max_probability
            
            return TransactionCategory.OTHER_EXPENSE, max_probability
        except Exception as e:
            print(f"⚠️ Ошибка при предсказании: {e}")
            description = str(row.get('RefNo', ''))
            amount = float(row.get('amount', 0))
            is_expense = bool(row.get('is_expense', 0))
            category = self._fallback_classify(description, amount, is_expense)
            return category, 0.5
    
    def _fallback_classify(self, description: str, amount: float, is_expense: bool) -> TransactionCategory:
        """Простая эвристическая классификация, если модель не обучена"""
        desc_lower = (description or "").lower()
        
        # Ключевые слова для категорий
        if any(word in desc_lower for word in ["food", "продукты", "магазин", "еда", "grocery"]):
            return TransactionCategory.PRODUCTS
        if any(word in desc_lower for word in ["transport", "транспорт", "метро", "такси", "uber"]):
            return TransactionCategory.TRANSPORT
        if any(word in desc_lower for word in ["rent", "аренда", "коммунальные"]):
            return TransactionCategory.UTILITIES
        if any(word in desc_lower for word in ["salary", "зарплата", "payroll"]):
            return TransactionCategory.SALARY
        if any(word in desc_lower for word in ["shopping", "покупки", "магазин"]):
            return TransactionCategory.CLOTHING
        
        # Большие суммы - возможно доход
        if not is_expense and amount > 10000:
            return TransactionCategory.SALARY
        
        return TransactionCategory.OTHER_EXPENSE
    
    def save_model(self):
        """Сохранение модели в файл"""
        if self.model is None:
            return
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        """Загрузка модели из файла"""
        if not os.path.exists(self.model_path):
            print(f"❌ Файл модели не существует: {self.model_path}")
            return
        
        try:
            print(f"📂 Загрузка модели из: {self.model_path}")
            # Игнорируем предупреждения о несовместимости версий
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
            
            # Проверяем формат данных (может быть словарь или кортеж)
            print(f"   Тип загруженных данных: {type(model_data)}")
            
            if isinstance(model_data, dict):
                # Новый формат - словарь
                print(f"   Формат: словарь (dict)")
                self.model = model_data.get('model')
                self.vectorizer = model_data.get('vectorizer')
                self.scaler = model_data.get('scaler')
                self.is_trained = model_data.get('is_trained', False)
            elif isinstance(model_data, tuple):
                # Старый формат - кортеж
                print(f"   Формат: кортеж (tuple), длина: {len(model_data)}")
                print(f"   Типы элементов: {[type(x).__name__ for x in model_data]}")
                
                # Пробуем разные варианты формата кортежа
                if len(model_data) >= 3:
                    # Вариант 1: (model, vectorizer, scaler)
                    self.model = model_data[0]
                    self.vectorizer = model_data[1]
                    self.scaler = model_data[2]
                    self.is_trained = True
                    print(f"   Использован формат: (model, vectorizer, scaler)")
                    
                    # Проверяем типы компонентов
                    from sklearn.preprocessing import LabelEncoder
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.preprocessing import StandardScaler
                    
                    if isinstance(self.vectorizer, LabelEncoder):
                        print(f"   ℹ️ Используется LabelEncoder для кодирования RefNo (категориальная переменная)")
                        print(f"   Модель поддерживает обработку новых RefNo через значение по умолчанию")
                    
                    if not isinstance(self.scaler, StandardScaler):
                        print(f"   ℹ️ Scaler имеет тип: {type(self.scaler)}, будет использоваться без масштабирования")
                elif len(model_data) == 2:
                    # Вариант 2: возможно (model, vectorizer) или что-то другое
                    print(f"   ⚠️ Кортеж из 2 элементов, пробуем интерпретировать...")
                    # Пробуем определить по типам
                    for i, item in enumerate(model_data):
                        item_type = type(item).__name__
                        if 'RandomForest' in item_type or 'Classifier' in item_type:
                            self.model = item
                            print(f"   Найден model в позиции {i}")
                        elif 'Tfidf' in item_type or 'Vectorizer' in item_type:
                            self.vectorizer = item
                            print(f"   Найден vectorizer в позиции {i}")
                        elif 'Scaler' in item_type:
                            self.scaler = item
                            print(f"   Найден scaler в позиции {i}")
                    self.is_trained = True
                else:
                    raise ValueError(f"Неожиданный формат кортежа: ожидалось 2-3 элемента, получено {len(model_data)}")
            else:
                print(f"   ⚠️ Неожиданный формат: {type(model_data)}")
                # Пробуем получить атрибуты напрямую, если это объект
                if hasattr(model_data, 'model'):
                    self.model = model_data.model
                if hasattr(model_data, 'vectorizer'):
                    self.vectorizer = model_data.vectorizer
                if hasattr(model_data, 'scaler'):
                    self.scaler = model_data.scaler
                if hasattr(model_data, 'is_trained'):
                    self.is_trained = model_data.is_trained
                else:
                    self.is_trained = self.model is not None and self.vectorizer is not None and self.scaler is not None
            
            # Проверка наличия всех компонентов
            if self.model is None:
                print(f"❌ Модель (classifier) не найдена в файле")
                self.is_trained = False
            elif self.vectorizer is None:
                print(f"❌ Vectorizer не найден в файле")
                self.is_trained = False
            elif self.scaler is None:
                print(f"❌ Scaler не найден в файле")
                self.is_trained = False
            else:
                print(f"✅ Все компоненты модели загружены успешно")
                print(f"   is_trained: {self.is_trained}")
                print(f"   model type: {type(self.model)}")
                print(f"   vectorizer type: {type(self.vectorizer)}")
                print(f"   scaler type: {type(self.scaler)}")
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.vectorizer = None
            self.scaler = None
            self.is_trained = False

