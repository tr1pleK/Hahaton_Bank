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
        '--test-size',
        type=float,
        default=0.2,
        help='Доля тестовой выборки (по умолчанию 0.2)'
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
        print(f"Ошибка: Файл {csv_path} не найден")
        sys.exit(1)
    
    print(f"Загрузка данных из {csv_path}...")
    
    # Создание и обучение классификатора
    classifier = TransactionClassifier()
    
    if args.force_retrain:
        print("⚠️ Режим принудительного переобучения включен")
        print("Начало обучения модели...")
    else:
        print("Проверка существования модели...")
        print("Начало обучения модели...")
    
    metrics = classifier.train(
        csv_path=str(csv_path),
        test_size=args.test_size,
        force_retrain=args.force_retrain
    )
    
    # Проверяем, была ли модель загружена из файла
    if metrics.get('loaded_from_file'):
        print("\n" + "="*50)
        print("Модель уже обучена!")
        print("="*50)
        print(f"Модель загружена из: {metrics['model_path']}")
        print(f"Для переобучения используйте флаг --force-retrain")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("Результаты обучения:")
        print("="*50)
        if 'accuracy' in metrics:
            print(f"Точность (Accuracy): {metrics['accuracy']:.4f}")
            print(f"Обучающих примеров: {metrics['train_samples']}")
            print(f"Тестовых примеров: {metrics['test_samples']}")
            print("\nОтчет по классификации:")
            print("-"*50)
            
            report = metrics.get('classification_report', {})
            for category, metrics_dict in report.items():
                if isinstance(metrics_dict, dict) and 'precision' in metrics_dict:
                    print(f"\n{category}:")
                    print(f"  Precision: {metrics_dict['precision']:.4f}")
                    print(f"  Recall: {metrics_dict['recall']:.4f}")
                    print(f"  F1-score: {metrics_dict['f1-score']:.4f}")
                    print(f"  Support: {metrics_dict['support']}")
        
        print("\n" + "="*50)
        print(f"Модель сохранена в: {classifier.model_path}")
        print("="*50)


if __name__ == "__main__":
    main()

