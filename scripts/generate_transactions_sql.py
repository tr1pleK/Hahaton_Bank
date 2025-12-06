"""
Скрипт для генерации SQL файла с транзакциями из ci_data.csv
Создает первые 300 записей транзакций для пользователя user@example.com (ID: 2)
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

# Маппинг категорий из CSV в TransactionCategory
CATEGORY_MAPPING = {
    'Food': 'Продукты',
    'Misc': 'Прочие расходы',
    'Rent': 'Коммунальные услуги',
    'Salary': 'Зарплата',
    'Shopping': 'Одежда',
}

def parse_date(date_str):
    """Парсит дату из различных форматов"""
    if pd.isna(date_str) or date_str == '':
        return None
    
    # Пробуем разные форматы
    formats = ['%d/%m/%Y', '%d/%m/%y', '%Y-%m-%d', '%d-%m-%Y', '%d-%m-%y']
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt).date()
        except:
            continue
    
    # Если ничего не подошло, пробуем pandas
    try:
        return pd.to_datetime(date_str, format="mixed", errors="coerce").date()
    except:
        return None

def generate_sql():
    """Генерирует SQL файл с транзакциями"""
    csv_path = Path(__file__).parent.parent / "app" / "ml" / "ci_data.csv"
    sql_path = Path(__file__).parent.parent / "init_transactions.sql"
    
    if not csv_path.exists():
        print(f"❌ Файл {csv_path} не найден!")
        return
    
    print(f"📖 Чтение CSV файла: {csv_path}")
    
    # Читаем CSV (пропускаем только заголовок, строка 2 - пустая, но мы её обработаем)
    df = pd.read_csv(csv_path, skiprows=1)
    # Удаляем пустые строки (где все значения NaN)
    df = df.dropna(how='all')
    
    # Очищаем данные
    df.columns = ["Date1", "Category", "RefNo", "Date2", "Withdrawal", "Deposit", "Balance"]
    
    # Фильтруем только валидные категории ПЕРЕД обработкой дат
    valid_cats = ["Food", "Misc", "Rent", "Salary", "Shopping"]
    df = df[df["Category"].isin(valid_cats)].copy()
    
    # Обрабатываем даты
    df["Date"] = pd.to_datetime(df["Date2"], format="mixed", errors="coerce")
    df["Withdrawal"] = pd.to_numeric(df["Withdrawal"], errors="coerce").fillna(0)
    df["Deposit"] = pd.to_numeric(df["Deposit"], errors="coerce").fillna(0)
    
    # Удаляем строки с пустыми датами
    df = df.dropna(subset=['Date']).copy()
    
    # Берем нужные колонки
    df = df[["Date", "Category", "Withdrawal", "Deposit", "Balance"]].copy()
    
    # Берем первые 300 записей
    df = df.head(300)
    
    print(f"✅ Обработано {len(df)} транзакций")
    
    # Генерируем SQL
    sql_lines = [
        "-- SQL файл для инициализации транзакций из ci_data.csv",
        "-- Автоматически сгенерирован из первых 300 записей",
        "-- Транзакции создаются для пользователя user@example.com (ID: 2)",
        "",
        "-- Удаляем существующие транзакции пользователя (опционально)",
        "-- DELETE FROM transactions WHERE user_id = 2;",
        "",
        "DO $$",
        "DECLARE",
        "    target_user_id INTEGER := 2;",
        "    transaction_count INTEGER := 0;",
        "BEGIN",
        "    -- Проверяем, что пользователь существует",
        "    IF NOT EXISTS (SELECT 1 FROM users WHERE id = target_user_id) THEN",
        "        RAISE EXCEPTION 'Пользователь с ID % не найден!', target_user_id;",
        "    END IF;",
        "",
        "    -- Вставляем транзакции",
    ]
    
    for idx, row in df.iterrows():
        date = row['Date']
        category = row['Category']
        withdrawal = float(row['Withdrawal']) if not pd.isna(row['Withdrawal']) else 0.0
        deposit = float(row['Deposit']) if not pd.isna(row['Deposit']) else 0.0
        
        # Определяем сумму и категорию
        if deposit > 0:
            amount = deposit
            mapped_category = CATEGORY_MAPPING.get(category, 'Прочие расходы')
        else:
            amount = withdrawal
            mapped_category = CATEGORY_MAPPING.get(category, 'Прочие расходы')
        
        # Пропускаем если дата невалидна
        if pd.isna(date):
            continue
        
        # Форматируем дату
        if isinstance(date, pd.Timestamp):
            date_str = date.strftime('%Y-%m-%d')
        else:
            parsed_date = parse_date(str(date))
            if parsed_date:
                date_str = parsed_date.strftime('%Y-%m-%d')
            else:
                continue
        
        # Экранируем категорию для SQL
        category_sql = mapped_category.replace("'", "''")
        
        # Создаем описание
        description = f"Транзакция {idx + 1}"
        description_sql = description.replace("'", "''")
        
        sql_lines.append(
            f"    INSERT INTO transactions (user_id, category, amount, description, date, created_at) "
            f"VALUES (target_user_id, '{category_sql}'::transactioncategory, {amount:.2f}, '{description_sql}', '{date_str}'::date, NOW());"
        )
        sql_lines.append("    transaction_count := transaction_count + 1;")
    
    sql_lines.extend([
        "",
        "    RAISE NOTICE 'Вставлено транзакций для пользователя %: %', target_user_id, transaction_count;",
        "END $$;",
        ""
    ])
    
    # Записываем SQL файл
    sql_content = "\n".join(sql_lines)
    sql_path.write_text(sql_content, encoding='utf-8')
    
    print(f"✅ SQL файл создан: {sql_path}")
    print(f"   Всего транзакций: {len(df)}")

if __name__ == "__main__":
    generate_sql()

