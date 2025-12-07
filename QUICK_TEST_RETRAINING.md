# Быстрый тест дообучения модели

## Самый простой способ

### 1. Через API (curl или Postman)

```bash
# 1. Авторизуйтесь
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "your_email", "password": "your_password"}'

# Сохраните токен из ответа

# 2. Создайте несколько транзакций
curl -X POST "http://localhost:8000/transactions" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2024-12-07",
    "amount": 5000,
    "description": "Тест дообучения",
    "category": "UTILITIES"
  }'

# 3. Запустите дообучение
curl -X POST "http://localhost:8000/ml/retrain?days_back=7" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Что проверить в ответе:**
- ✅ `"success": true`
- ✅ `"new_transactions_count" > 0`
- ✅ Есть метрики `f1_weighted` и `f1_macro`

### 2. Через Python скрипт

```bash
# 1. Установите зависимости (если еще не установлены)
pip install requests

# 2. Отредактируйте test_retraining.py:
#    - Измените EMAIL и PASSWORD на ваши
#    - При необходимости измените BASE_URL

# 3. Запустите скрипт
python test_retraining.py
```

Скрипт автоматически:
- ✅ Авторизуется
- ✅ Проверит модель до дообучения
- ✅ Создаст тестовые транзакции
- ✅ Запустит дообучение
- ✅ Проверит модель после дообучения
- ✅ Протестирует предсказания

### 3. Проверка файла модели

```bash
# До дообучения
ls -lh app/ml/classifier_v2.pkl
stat app/ml/classifier_v2.pkl

# Запустите дообучение через API

# После дообучения
ls -lh app/ml/classifier_v2.pkl
stat app/ml/classifier_v2.pkl
```

**Ожидаемый результат:** Время модификации файла должно измениться.

## Критерии успеха

✅ **Модель дообучилась, если:**
1. API вернул `"success": true`
2. `new_transactions_count > 0`
3. Файл модели обновился (время модификации изменилось)
4. Метрики F1 в разумных пределах (0.7-1.0)

## Частые проблемы

**Проблема:** `new_transactions_count: 0`
- **Решение:** Создайте транзакции с датами в пределах последних 7 дней

**Проблема:** Ошибка авторизации
- **Решение:** Проверьте email и password, убедитесь что сервер запущен

**Проблема:** Таймаут при дообучении
- **Решение:** Увеличьте таймаут или уменьшите `days_back`

## Подробная документация

См. `TEST_MODEL_RETRAINING.md` для детального руководства.

