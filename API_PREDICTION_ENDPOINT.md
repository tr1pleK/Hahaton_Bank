# API Endpoint для предсказания категории транзакции

## Описание

Endpoint `/transactions/predict-category` позволяет получить предсказание категории транзакции с использованием обученной ML модели. 

**Процесс работы:**
1. Принимает данные с фронтенда в формате CSV (Date, RefNo, Withdrawal, Deposit, Balance)
2. Преобразует их во внутренний DataFrame, как при чтении из CSV
3. Применяет логику обработки признаков (как при обучении модели)
4. Выполняет предсказание
5. Возвращает результат в JSON с предсказанной категорией и вероятностью

## Endpoint

```
POST /transactions/predict-category
```

## Аутентификация

Требуется авторизация через Bearer токен:
```
Authorization: Bearer YOUR_ACCESS_TOKEN
```

## Запрос

### Тело запроса (JSON)

Endpoint принимает **массив транзакций** (минимум 1 элемент):

```json
[
  {
    "Date": "1/1/2023",
    "RefNo": "3.00E+11",
    "Withdrawal": 3950.0,
    "Deposit": 0.0,
    "Balance": 1837.23
  },
  {
    "Date": "3/1/2023",
    "RefNo": "3.37E+11",
    "Withdrawal": 0.0,
    "Deposit": 55.0,
    "Balance": 1787.23
  }
]
```

### Параметры транзакции

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `Date` | string | Да | Дата транзакции (формат: DD/MM/YYYY или MM/DD/YYYY) |
| `RefNo` | string | Да | Описание/идентификатор транзакции |
| `Withdrawal` | float | Нет | Сумма снятия (расход). По умолчанию: 0.0 |
| `Deposit` | float | Нет | Сумма пополнения (доход). По умолчанию: 0.0 |
| `Balance` | float | Нет | Баланс после транзакции (не используется для предсказания) |

## Ответ

### Успешный ответ (200 OK)

Возвращается **массив результатов** в том же порядке, что и запрос:

```json
[
  {
    "Date": "1/1/2023",
    "RefNo": "3.00E+11",
    "Withdrawal": 3950.0,
    "Deposit": 0.0,
    "Balance": 1837.23,
    "Category": "Коммунальные услуги",
    "Probability": 0.8523
  },
  {
    "Date": "3/1/2023",
    "RefNo": "3.37E+11",
    "Withdrawal": 0.0,
    "Deposit": 55.0,
    "Balance": 1787.23,
    "Category": "Прочие расходы",
    "Probability": 0.7234
  }
]
```

### Поля ответа (для каждой транзакции)

| Поле | Тип | Описание |
|------|-----|----------|
| `Date` | string | Дата транзакции (та же, что в запросе) |
| `RefNo` | string | Описание транзакции (то же, что в запросе) |
| `Withdrawal` | float | Сумма снятия (та же, что в запросе) |
| `Deposit` | float | Сумма пополнения (та же, что в запросе) |
| `Balance` | float | Баланс (тот же, что в запросе) |
| `Category` | string | Предсказанная категория транзакции |
| `Probability` | float | Вероятность предсказания (от 0.0 до 1.0) |

**Важно:** Порядок транзакций в ответе соответствует порядку в запросе.

### Возможные категории

**Расходы:**
- `Продукты` - покупка продуктов питания
- `Транспорт` - транспортные расходы
- `Кафе и рестораны` - расходы на кафе и рестораны
- `Здоровье` - медицинские расходы
- `Развлечения` - развлекательные расходы
- `Одежда` - покупка одежды
- `Коммунальные услуги` - коммунальные платежи, аренда
- `Образование` - образовательные расходы
- `Подарки` - расходы на подарки
- `Прочие расходы` - прочие расходы

**Доходы:**
- `Зарплата` - заработная плата
- `Премия` - премии
- `Инвестиции` - доходы от инвестиций
- `Подарок` - полученные подарки
- `Прочие доходы` - прочие доходы

## Примеры использования

### Пример 1: Предсказание категории для одной транзакции

```bash
curl -X POST "http://localhost:8000/transactions/predict-category" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "Date": "1/1/2023",
      "RefNo": "3.00E+11",
      "Withdrawal": 3950.0,
      "Deposit": 0.0,
      "Balance": 1837.23
    }
  ]'
```

**Ответ:**
```json
[
  {
    "Date": "1/1/2023",
    "RefNo": "3.00E+11",
    "Withdrawal": 3950.0,
    "Deposit": 0.0,
    "Balance": 1837.23,
    "Category": "Коммунальные услуги",
    "Probability": 0.8234
  }
]
```

### Пример 2: Предсказание категорий для массива транзакций

```bash
curl -X POST "http://localhost:8000/transactions/predict-category" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "Date": "25/01/23",
      "RefNo": "CHASR23023738798",
      "Withdrawal": 0.0,
      "Deposit": 34800.0,
      "Balance": 36119.74
    },
    {
      "Date": "3/1/2023",
      "RefNo": "3.00E+11",
      "Withdrawal": 56.0,
      "Deposit": 0.0,
      "Balance": 1731.23
    },
    {
      "Date": "15/03/26",
      "RefNo": "3.06E+11",
      "Withdrawal": 350.0,
      "Deposit": 0.0,
      "Balance": 21327.0
    }
  ]'
```

**Ответ:**
```json
[
  {
    "Date": "25/01/23",
    "RefNo": "CHASR23023738798",
    "Withdrawal": 0.0,
    "Deposit": 34800.0,
    "Balance": 36119.74,
    "Category": "Зарплата",
    "Probability": 0.9567
  },
  {
    "Date": "3/1/2023",
    "RefNo": "3.00E+11",
    "Withdrawal": 56.0,
    "Deposit": 0.0,
    "Balance": 1731.23,
    "Category": "Продукты",
    "Probability": 0.7821
  },
  {
    "Date": "15/03/26",
    "RefNo": "3.06E+11",
    "Withdrawal": 350.0,
    "Deposit": 0.0,
    "Balance": 21327.0,
    "Category": "Прочие расходы",
    "Probability": 0.8041
  }
]
```

## Ошибки

### 401 Unauthorized
```json
{
  "detail": "Не авторизован"
}
```

### 400 Bad Request
```json
{
  "detail": "Массив транзакций не может быть пустым"
}
```
или
```json
{
  "detail": "Ошибка валидации данных"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Ошибка при предсказании категории: <описание ошибки>"
}
```

## Примечания

1. **Модель**: Endpoint использует обученную ML модель, сохраненную в `ml_models/transaction_classifier.pkl`
2. **Обработка данных**: Данные обрабатываются точно так же, как при обучении модели:
   - Преобразуются в DataFrame
   - Применяется та же логика извлечения признаков
   - Используются те же трансформеры (TF-IDF, StandardScaler)
3. **Fallback**: Если модель не обучена или не найдена, используется простая эвристическая классификация с вероятностью 0.5
4. **Вероятность**: Вероятность предсказания округляется до 4 знаков после запятой
5. **Формат данных**: Формат данных в ответе совпадает с форматом запроса, но с добавленными полями `Category` и `Probability`
6. **Дата**: Поддерживаются различные форматы даты (DD/MM/YYYY, MM/DD/YYYY, DD/MM/YY и т.д.)

## Интеграция с фронтендом

Фронтенд может использовать этот endpoint для:
- Автоматического определения категории при вводе транзакции
- Отображения вероятности предсказания для пользователя
- Предложения категории пользователю с возможностью ручной корректировки

Пример использования в JavaScript:

```javascript
async function predictCategories(transactions) {
  const response = await fetch('/transactions/predict-category', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(transactions)
  });
  
  const results = await response.json();
  return results;
}

// Пример использования с одной транзакцией
const singleTransaction = [{
  Date: "1/1/2023",
  RefNo: "3.00E+11",
  Withdrawal: 3950.0,
  Deposit: 0.0,
  Balance: 1837.23
}];

const results = await predictCategories(singleTransaction);
console.log(`Категория: ${results[0].Category}`);
console.log(`Вероятность: ${results[0].Probability}`);

// Пример использования с массивом транзакций
const multipleTransactions = [
  {
    Date: "1/1/2023",
    RefNo: "3.00E+11",
    Withdrawal: 3950.0,
    Deposit: 0.0,
    Balance: 1837.23
  },
  {
    Date: "15/03/26",
    RefNo: "3.06E+11",
    Withdrawal: 350.0,
    Deposit: 0.0,
    Balance: 21327.0
  }
];

const allResults = await predictCategories(multipleTransactions);
allResults.forEach((result, index) => {
  console.log(`Транзакция ${index + 1}: ${result.Category} (${result.Probability})`);
});
```

