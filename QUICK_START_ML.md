# Быстрый старт: Обучение модели классификации транзакций

## Шаг 1: Установка зависимостей

```bash
pip install -r requirements.txt
```

## Шаг 2: Обучение модели

Используйте предоставленный CSV файл для обучения:

```bash
python scripts/train_transaction_classifier.py "c:\Users\Егор\Downloads\Telegram Desktop\ci_data.csv"
```

## Шаг 3: Проверка работы

После обучения модель автоматически сохранится в `ml_models/transaction_classifier.pkl` и будет использоваться при создании новых транзакций через API.

## Пример использования через API

### Создание транзакции (автоматическая классификация)

```http
POST /transactions
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN

{
  "amount": 100.0,
  "description": "3.00E+11",
  "date": "2024-01-15"
}
```

Модель автоматически определит категорию на основе описания и суммы.

### Ручная корректировка категории

```http
PUT /transactions/{transaction_id}
Content-Type: application/json
Authorization: Bearer YOUR_TOKEN

{
  "category": "Продукты"
}
```

## Структура данных CSV

CSV файл должен содержать следующие колонки:
- `Date` - Дата (формат: DD/MM/YYYY или MM/DD/YYYY)
- `Category` - Категория (Rent, Misc, Food, Salary, Shopping, Transport)
- `RefNo` - Описание транзакции
- `Withdrawal` - Сумма расхода
- `Deposit` - Сумма дохода
- `Balance` - Баланс (не используется для обучения)

## Результаты обучения

После обучения вы увидите:
- Точность модели (Accuracy)
- Метрики для каждой категории (Precision, Recall, F1-score)
- Количество обучающих и тестовых примеров

## Примечания

- Модель автоматически загружается при первом использовании
- Если модель не обучена, используется простая эвристическая классификация
- Ручная корректировка категорий не влияет на модель (можно добавить дообучение в будущем)

