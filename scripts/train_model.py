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
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Принудительно переобучить модель, даже если файл уже существует'
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
    
    # Проверка существования модели
    if not args.force_retrain and classifier.is_trained:
        print(f"📦 Модель уже обучена по пути: {classifier.model_path}")
        print(f"   Для переобучения используйте флаг --force-retrain")
        print(f"✅ Используется существующая модель")
        return
    
    print("🚀 Начало обучения модели...")
    metrics = classifier.train(csv_path=str(csv_path))
    
    print("\n" + "="*50)
    print("✅ Результаты обучения:")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("="*50)
    print(f"💾 Модель сохранена в: {classifier.model_path}")
    print("="*50)


if __name__ == "__main__":
    main()

