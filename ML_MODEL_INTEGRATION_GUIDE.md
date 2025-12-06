# Руководство по внедрению новой ML модели

## Обзор

Это руководство поможет вам заменить старую модель машинного обучения на новую, обученную другим алгоритмом.

## Структура файлов

```
backend/
├── app/
│   └── ml/
│       ├── categorizer.py          # Интерфейс для категоризации (используется API)
│       ├── transaction_classifier.py  # НОВЫЙ: Ваш новый классификатор
│       └── model_loader.py         # Загрузка сохраненной модели
├── scripts/
│   └── train_model.py              # НОВЫЙ: Скрипт для обучения модели
├── ml_models/                      # Директория для сохраненных моделей
│   └── transaction_classifier.pkl  # Обученная модель (создается после обучения)
└── requirements.txt                # Зависимости (может потребоваться обновление)
```

## Шаг 1: Подготовка кода обучения

### 1.1 Создайте файл `app/ml/transaction_classifier.py`

Вставьте сюда код обучения вашей модели с другого ПК. Структура должна быть примерно такой:

```python
"""Классификатор транзакций - ваша новая модель"""
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

# Импорты ваших библиотек для ML
# Например:
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# или ваши библиотеки

from app.models.category import TransactionCategory

# Маппинг категорий из ваших данных в категории системы
CATEGORY_MAPPING = {
    'Rent': TransactionCategory.UTILITIES,
    'Misc': TransactionCategory.OTHER_EXPENSE,
    'Food': TransactionCategory.PRODUCTS,
    'Salary': TransactionCategory.SALARY,
    'Shopping': TransactionCategory.CLOTHING,
    'Transport': TransactionCategory.TRANSPORT,
}


class TransactionClassifier:
    """Ваш новый классификатор транзакций"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация классификатора
        
        Args:
            model_path: Путь к сохраненной модели (опционально)
        """
        # Определите путь к модели
        if model_path:
            self.model_path = model_path
        else:
            # Путь по умолчанию
            default_path = Path(__file__).parent.parent.parent / "ml_models" / "transaction_classifier.pkl"
            self.model_path = str(default_path)
        
        # Инициализируйте переменные модели
        self.model = None
        self.is_trained = False
        
        # Загрузите модель, если она существует
        if os.path.exists(self.model_path):
            print(f"🔍 Найдена модель по пути: {self.model_path}")
            self.load_model()
        else:
            print(f"⚠️ Модель не найдена по пути: {self.model_path}")
    
    def train(self, csv_path: str, **kwargs) -> Dict[str, Any]:
        """
        Обучение модели на данных из CSV
        
        Args:
            csv_path: Путь к CSV файлу с данными
            **kwargs: Дополнительные параметры для обучения
            
        Returns:
            Словарь с метриками обучения
        """
        # ВСТАВЬТЕ СЮДА ВАШ КОД ОБУЧЕНИЯ
        
        # Пример структуры:
        # 1. Загрузка данных
        # df = pd.read_csv(csv_path)
        
        # 2. Предобработка данных
        # df = self._preprocess_data(df)
        
        # 3. Извлечение признаков
        # X, y = self._extract_features(df)
        
        # 4. Обучение модели
        # self.model = YourModel()
        # self.model.fit(X, y)
        
        # 5. Оценка модели
        # metrics = self._evaluate_model(X_test, y_test)
        
        # 6. Сохранение модели
        # self.save_model()
        
        # 7. Возврат метрик
        return {
            'accuracy': 0.0,  # Замените на реальные метрики
            'is_trained': True
        }
    
    def predict(self, description: str, amount: float, is_expense: bool = True, 
                date: Optional[datetime] = None) -> Tuple[TransactionCategory, float]:
        """
        Предсказание категории для одной транзакции
        
        Args:
            description: Описание транзакции (RefNo)
            amount: Сумма транзакции
            is_expense: Является ли транзакция расходом
            date: Дата транзакции
            
        Returns:
            Tuple с категорией и вероятностью
        """
        if not self.is_trained or self.model is None:
            # Fallback классификация
            return TransactionCategory.OTHER_EXPENSE, 0.5
        
        # ВСТАВЬТЕ СЮДА ВАШ КОД ПРЕДСКАЗАНИЯ
        
        # Пример:
        # 1. Подготовка признаков
        # features = self._prepare_features(description, amount, is_expense, date)
        
        # 2. Предсказание
        # prediction = self.model.predict(features)
        # probability = self.model.predict_proba(features)
        
        # 3. Преобразование в TransactionCategory
        # category = self._map_to_category(prediction)
        
        return TransactionCategory.OTHER_EXPENSE, 0.5
    
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
        
        # ВСТАВЬТЕ СЮДА ВАШ КОД ПРЕДСКАЗАНИЯ ИЗ DATAFRAME
        
        # Пример:
        # 1. Обработка DataFrame (как при обучении)
        # processed_df = self._preprocess_dataframe(df)
        
        # 2. Извлечение признаков
        # features = self._extract_features_from_df(processed_df)
        
        # 3. Предсказание
        # prediction = self.model.predict(features)
        # probability = self.model.predict_proba(features)
        
        return TransactionCategory.OTHER_EXPENSE, 0.5
    
    def save_model(self):
        """Сохранение модели в файл"""
        if self.model is None:
            return
        
        # ВСТАВЬТЕ СЮДА ВАШ КОД СОХРАНЕНИЯ МОДЕЛИ
        
        # Пример с pickle:
        # model_data = {
        #     'model': self.model,
        #     'is_trained': self.is_trained,
        #     'trained_at': datetime.now().isoformat()
        # }
        # 
        # os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        # with open(self.model_path, 'wb') as f:
        #     pickle.dump(model_data, f)
        
        print(f"✅ Модель сохранена в: {self.model_path}")
    
    def load_model(self):
        """Загрузка модели из файла"""
        if not os.path.exists(self.model_path):
            print(f"❌ Файл модели не существует: {self.model_path}")
            return
        
        try:
            # ВСТАВЬТЕ СЮДА ВАШ КОД ЗАГРУЗКИ МОДЕЛИ
            
            # Пример с pickle:
            # with open(self.model_path, 'rb') as f:
            #     model_data = pickle.load(f)
            # 
            # self.model = model_data.get('model')
            # self.is_trained = model_data.get('is_trained', False)
            
            print(f"✅ Модель загружена из: {self.model_path}")
        except Exception as e:
            print(f"❌ Ошибка при загрузке модели: {e}")
            self.is_trained = False
```

### 1.2 Создайте скрипт обучения `scripts/train_model.py`

```python
"""Скрипт для обучения модели классификации транзакций"""
import sys
from pathlib import Path

# Добавляем корневую директорию проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.ml.transaction_classifier import TransactionClassifier


def main():
    """Основная функция для обучения модели"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение модели классификации транзакций')
    parser.add_argument(
        'csv_path',
        type=str,
        help='Путь к CSV файлу с тренировочными данными'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Путь для сохранения модели (по умолчанию: ml_models/transaction_classifier.pkl)'
    )
    
    args = parser.parse_args()
    
    # Проверка существования файла
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"❌ Ошибка: Файл {csv_path} не найден")
        sys.exit(1)
    
    print(f"📂 Загрузка данных из {csv_path}...")
    
    # Создание классификатора
    classifier = TransactionClassifier(model_path=args.model_path)
    
    print("🚀 Начало обучения модели...")
    metrics = classifier.train(csv_path=str(csv_path))
    
    print("\n" + "="*50)
    print("✅ Результаты обучения:")
    print("="*50)
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print("="*50)
    print(f"💾 Модель сохранена в: {classifier.model_path}")
    print("="*50)


if __name__ == "__main__":
    main()
```

## Шаг 2: Обновление зависимостей

### 2.1 Обновите `requirements.txt`

Добавьте все необходимые библиотеки для вашей новой модели:

```txt
# Существующие зависимости
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0
email-validator==2.1.0
python-multipart==0.0.6
passlib[bcrypt]==1.7.4
bcrypt==4.0.1
python-jose[cryptography]==3.3.0

# ML зависимости (обновите версии под вашу модель)
scikit-learn==1.3.2  # или ваша версия
pandas==2.1.3        # или ваша версия
numpy==1.26.2         # или ваша версия

# ДОБАВЬТЕ СЮДА ВСЕ ДОПОЛНИТЕЛЬНЫЕ БИБЛИОТЕКИ
# Например:
# xgboost==1.7.0
# lightgbm==3.3.0
# catboost==1.1.0
# tensorflow==2.13.0
# torch==2.0.0
# и т.д.
```

## Шаг 3: Обновление categorizer.py

Обновите `app/ml/categorizer.py` для использования новой модели:

```python
"""Категоризатор транзакций с использованием ML модели"""
from typing import Optional
from datetime import datetime
from app.models.category import TransactionCategory
from app.ml.transaction_classifier import TransactionClassifier

# Глобальный экземпляр классификатора (загружается один раз)
_classifier: Optional[TransactionClassifier] = None


def _get_classifier() -> TransactionClassifier:
    """Получить или создать экземпляр классификатора"""
    global _classifier
    if _classifier is None:
        _classifier = TransactionClassifier()
    return _classifier


def categorize_transaction(
    description: str, 
    amount: float, 
    is_expense: bool = True,
    date: Optional[datetime] = None
) -> TransactionCategory:
    """
    Определяет категорию транзакции с использованием ML модели
    
    Args:
        description: Описание транзакции
        amount: Сумма транзакции
        is_expense: Является ли транзакция расходом
        date: Дата транзакции
        
    Returns:
        Категория транзакции
    """
    classifier = _get_classifier()
    category, probability = classifier.predict(
        description=description,
        amount=amount,
        is_expense=is_expense,
        date=date
    )
    return category
```

## Шаг 4: Добавление API endpoint для предсказаний (опционально)

Если нужен endpoint для предсказаний, добавьте в `app/api/transactions.py`:

```python
from app.ml.transaction_classifier import TransactionClassifier
from app.schemas.transaction import CSVTransactionInput, CSVTransactionPredictionResponse

@router.post("/predict-category", response_model=List[CSVTransactionPredictionResponse])
async def predict_category_endpoint(
    transactions: List[CSVTransactionInput],
    current_user: User = Depends(get_current_user)
) -> List[CSVTransactionPredictionResponse]:
    """Предсказание категорий для массива транзакций"""
    try:
        classifier = TransactionClassifier()
        
        results = []
        for transaction in transactions:
            # Преобразование в DataFrame
            df = pd.DataFrame([{
                'Date': transaction.Date,
                'RefNo': transaction.RefNo,
                'Withdrawal': transaction.Withdrawal or 0.0,
                'Deposit': transaction.Deposit or 0.0,
                'Balance': transaction.Balance
            }])
            
            # Предсказание
            category, probability = classifier.predict_from_dataframe(df)
            
            results.append(CSVTransactionPredictionResponse(
                Date=transaction.Date,
                RefNo=transaction.RefNo,
                Withdrawal=transaction.Withdrawal,
                Deposit=transaction.Deposit,
                Balance=transaction.Balance,
                Category=category.value,
                Probability=round(probability, 4)
            ))
        
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при предсказании: {str(e)}"
        )
```

## Шаг 5: Обучение модели

### 5.1 Установите зависимости

```bash
pip install -r requirements.txt
```

### 5.2 Обучите модель

```bash
python scripts/train_model.py "путь/к/вашему/файлу/ci_data.csv"
```

Модель будет сохранена в `ml_models/transaction_classifier.pkl`

## Шаг 6: Проверка работы

### 6.1 Локальная проверка

```bash
python -c "from app.ml.transaction_classifier import TransactionClassifier; c = TransactionClassifier(); print('Модель загружена:', c.is_trained)"
```

### 6.2 Запуск сервера

```bash
# Локально
uvicorn app.main:app --reload

# Или через Docker
docker-compose up --build
```

## Шаг 7: Обновление Docker (если нужно)

### 7.1 Обновите docker-compose.yml

Убедитесь, что директория `ml_models` правильно монтируется:

```yaml
volumes:
  - .:/app
  - ./ml_models:/app/ml_models  # Изменено с ml_models:/app/ml_models
```

### 7.2 Пересоберите Docker образ

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

## Чеклист внедрения

- [ ] Создан файл `app/ml/transaction_classifier.py` с вашим кодом обучения
- [ ] Создан скрипт `scripts/train_model.py` для обучения
- [ ] Обновлен `requirements.txt` с необходимыми библиотеками
- [ ] Обновлен `app/ml/categorizer.py` для использования новой модели
- [ ] Модель обучена и сохранена в `ml_models/transaction_classifier.pkl`
- [ ] Модель загружается при старте приложения
- [ ] API endpoint работает корректно (если добавлен)
- [ ] Протестировано локально
- [ ] Протестировано в Docker (если используется)

## Решение проблем

### Модель не загружается
- Проверьте путь к файлу модели
- Убедитесь, что файл существует и доступен
- Проверьте формат сохранения/загрузки модели

### Ошибки при обучении
- Проверьте формат входных данных (CSV)
- Убедитесь, что все зависимости установлены
- Проверьте версии библиотек

### Ошибки при предсказании
- Убедитесь, что формат входных данных совпадает с форматом при обучении
- Проверьте, что все признаки извлекаются корректно

## Дополнительные советы

1. **Версионирование моделей**: Сохраняйте модели с версиями (например, `transaction_classifier_v2.pkl`)
2. **Логирование**: Добавьте логирование для отслеживания работы модели
3. **Метрики**: Сохраняйте метрики обучения для сравнения моделей
4. **Тестирование**: Создайте тесты для проверки работы модели

## Поддержка

Если возникнут проблемы, проверьте:
- Логи приложения
- Формат данных модели
- Совместимость версий библиотек
- Структуру входных данных

