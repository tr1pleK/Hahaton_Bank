#!/usr/bin/env python3
"""
Скрипт для тестирования дообучения модели классификации транзакций
"""
import requests
import time
import sys
from pathlib import Path
from datetime import date, timedelta

# Настройки
BASE_URL = "http://localhost:8000"
EMAIL = "demo@example.com"  # Измените на ваш email
PASSWORD = "demo123"  # Измените на ваш пароль

def print_step(step_num, description):
    """Вывод шага тестирования"""
    print(f"\n{'='*60}")
    print(f"Шаг {step_num}: {description}")
    print(f"{'='*60}")

def test_retraining():
    """Основная функция тестирования"""
    
    print("🧪 Тестирование дообучения модели классификации транзакций")
    print(f"🌐 Базовый URL: {BASE_URL}")
    
    # Шаг 1: Авторизация
    print_step(1, "Авторизация")
    try:
        login_response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"email": EMAIL, "password": PASSWORD},
            timeout=10
        )
        login_response.raise_for_status()
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print(f"✅ Авторизация успешна")
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка авторизации: {e}")
        print(f"   Убедитесь, что сервер запущен и учетные данные верны")
        return False
    
    # Шаг 2: Проверка времени модификации модели ДО
    print_step(2, "Проверка модели ДО дообучения")
    model_path = Path("app/ml/classifier_v2.pkl")
    mtime_before = None
    
    if model_path.exists():
        mtime_before = model_path.stat().st_mtime
        size_before = model_path.stat().st_size
        print(f"✅ Модель найдена")
        print(f"   Путь: {model_path.absolute()}")
        print(f"   Время модификации: {time.ctime(mtime_before)}")
        print(f"   Размер: {size_before / 1024:.2f} KB")
    else:
        print(f"⚠️  Модель не найдена по пути: {model_path}")
        print(f"   Модель будет создана при дообучении")
    
    # Шаг 3: Создание тестовых транзакций
    print_step(3, "Создание тестовых транзакций")
    today = date.today()
    test_transactions = [
        {
            "date": (today - timedelta(days=2)).isoformat(),
            "amount": 5000,
            "description": "Аренда квартиры (тест дообучения)",
            "category": "UTILITIES"
        },
        {
            "date": (today - timedelta(days=1)).isoformat(),
            "amount": 1500,
            "description": "Покупка продуктов (тест дообучения)",
            "category": "PRODUCTS"
        },
        {
            "date": today.isoformat(),
            "amount": 2000,
            "description": "Покупка одежды (тест дообучения)",
            "category": "CLOTHING"
        }
    ]
    
    created_count = 0
    for i, txn in enumerate(test_transactions, 1):
        try:
            response = requests.post(
                f"{BASE_URL}/transactions",
                json=txn,
                headers=headers,
                timeout=10
            )
            if response.status_code == 200 or response.status_code == 201:
                print(f"  ✅ Транзакция {i} создана: {txn['description']}")
                created_count += 1
            else:
                print(f"  ⚠️  Транзакция {i} не создана: {response.status_code} - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Ошибка при создании транзакции {i}: {e}")
    
    if created_count == 0:
        print("⚠️  Не удалось создать тестовые транзакции")
        print("   Продолжаем тест с существующими транзакциями...")
    
    # Шаг 4: Запуск дообучения
    print_step(4, "Запуск дообучения модели")
    try:
        print("   Отправка запроса на дообучение...")
        retrain_response = requests.post(
            f"{BASE_URL}/ml/retrain?days_back=7",
            headers=headers,
            timeout=300  # 5 минут на дообучение
        )
        retrain_response.raise_for_status()
        result = retrain_response.json()
        
        if result.get("success"):
            print("✅ Дообучение завершено успешно!")
            print(f"\n📊 Результаты:")
            print(f"   Новых транзакций: {result.get('new_transactions_count', 0)}")
            
            metrics = result.get("metrics", {})
            if metrics:
                print(f"   F1 (weighted): {metrics.get('f1_weighted', 'N/A'):.4f}" if isinstance(metrics.get('f1_weighted'), (int, float)) else f"   F1 (weighted): {metrics.get('f1_weighted', 'N/A')}")
                print(f"   F1 (macro): {metrics.get('f1_macro', 'N/A'):.4f}" if isinstance(metrics.get('f1_macro'), (int, float)) else f"   F1 (macro): {metrics.get('f1_macro', 'N/A')}")
                print(f"   Размер обучающей выборки: {metrics.get('train_size', 'N/A')}")
                print(f"   Размер тестовой выборки: {metrics.get('test_size', 'N/A')}")
                print(f"   Всего данных: {metrics.get('total_samples', 'N/A')}")
            
            print(f"   Путь к модели: {result.get('model_path', 'N/A')}")
        else:
            print(f"❌ Дообучение не удалось: {result.get('message', 'Unknown error')}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Таймаут при дообучении (превышено 5 минут)")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Ошибка при дообучении: {e}")
        return False
    
    # Шаг 5: Проверка времени модификации модели ПОСЛЕ
    print_step(5, "Проверка модели ПОСЛЕ дообучения")
    if model_path.exists():
        mtime_after = model_path.stat().st_mtime
        size_after = model_path.stat().st_size
        print(f"✅ Модель найдена")
        print(f"   Время модификации: {time.ctime(mtime_after)}")
        print(f"   Размер: {size_after / 1024:.2f} KB")
        
        if mtime_before:
            if mtime_after > mtime_before:
                print(f"\n✅ Модель была обновлена!")
                print(f"   Разница во времени: {mtime_after - mtime_before:.2f} секунд")
                if size_after != size_before:
                    size_diff = size_after - size_before
                    print(f"   Изменение размера: {size_diff:+d} байт ({size_diff/1024:+.2f} KB)")
            else:
                print(f"\n⚠️  Время модификации не изменилось")
                print(f"   Возможно, модель не была пересохранена")
        else:
            print(f"\n✅ Модель была создана!")
    else:
        print(f"❌ Модель не найдена после дообучения")
        return False
    
    # Шаг 6: Тест предсказания
    print_step(6, "Тест предсказания категории")
    try:
        test_prediction = {
            "date": today.isoformat(),
            "description": "Покупка продуктов в магазине",
            "withdrawal": 2500,
            "deposit": 0,
            "balance": 10000
        }
        
        predict_response = requests.post(
            f"{BASE_URL}/transactions/predict",
            json=[test_prediction],
            headers=headers,
            timeout=10
        )
        predict_response.raise_for_status()
        predictions = predict_response.json()
        
        if predictions and len(predictions) > 0:
            pred = predictions[0]
            print(f"✅ Предсказание получено:")
            print(f"   Категория: {pred.get('category', 'N/A')}")
            print(f"   Вероятность: {pred.get('category_probability', 'N/A'):.4f}" if isinstance(pred.get('category_probability'), (int, float)) else f"   Вероятность: {pred.get('category_probability', 'N/A')}")
        else:
            print("⚠️  Предсказание не получено")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Ошибка при получении предсказания: {e}")
    
    # Итоги
    print(f"\n{'='*60}")
    print("✅ Тестирование завершено!")
    print(f"{'='*60}")
    
    return True

if __name__ == "__main__":
    # Проверка аргументов командной строки
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
    if len(sys.argv) > 2:
        EMAIL = sys.argv[2]
    if len(sys.argv) > 3:
        PASSWORD = sys.argv[3]
    
    print(f"Использование: python test_retraining.py [BASE_URL] [EMAIL] [PASSWORD]")
    print(f"Текущие настройки:")
    print(f"  BASE_URL: {BASE_URL}")
    print(f"  EMAIL: {EMAIL}")
    print(f"  PASSWORD: {'*' * len(PASSWORD)}")
    print(f"\nДля изменения настроек отредактируйте переменные в начале скрипта")
    print(f"или передайте их как аргументы командной строки.\n")
    
    success = test_retraining()
    sys.exit(0 if success else 1)

