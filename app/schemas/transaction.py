"""Схемы для валидации транзакций"""
from pydantic import BaseModel, Field, field_validator, ValidationError
from datetime import date, datetime, timedelta
from typing import Optional, List, Union


class TransactionCreate(BaseModel):
    """Схема для создания транзакции"""
    amount: float = Field(..., gt=0)
    description: Optional[str] = None
    date: date
    category: Optional[str] = None
    
    @field_validator('date', mode='before')
    @classmethod
    def validate_date(cls, v):
        """Валидация даты транзакции"""
        # Если строка, пытаемся преобразовать
        if isinstance(v, str):
            v = v.strip()
            try:
                v = datetime.strptime(v, '%Y-%m-%d').date()
            except ValueError:
                raise ValueError(
                    "Неверный формат даты. Используйте формат: YYYY-MM-DD (например: 2023-01-15)"
                )
        
        # Проверяем, что это объект date
        if not isinstance(v, date):
            raise ValueError("Поле 'date' должно быть датой")
        
        # Не разрешаем будущие даты
        today = date.today()
        if v > today:
            raise ValueError(
                f"Дата транзакции не может быть в будущем. Получено: {v.strftime('%Y-%m-%d')}, "
                f"сегодня: {today.strftime('%Y-%m-%d')}"
            )
        
        # Не разрешаем слишком старые даты (старше 10 лет)
        min_date = today - timedelta(days=3650)  # 10 лет назад
        if v < min_date:
            raise ValueError(
                f"Дата транзакции слишком старая. Минимальная дата: {min_date.strftime('%Y-%m-%d')}, "
                f"получено: {v.strftime('%Y-%m-%d')}"
            )
        
        return v


class TransactionUpdate(BaseModel):
    """Схема для обновления транзакции"""
    amount: Optional[float] = Field(None, gt=0)
    description: Optional[str] = None
    date: Optional[date] = None
    category: Optional[str] = None
    
    @field_validator('date', mode='before')
    @classmethod
    def validate_date(cls, v):
        """Валидация даты транзакции (опциональное поле)"""
        if v is None:
            return None
        
        # Если строка, пытаемся преобразовать
        if isinstance(v, str):
            v = v.strip()
            try:
                v = datetime.strptime(v, '%Y-%m-%d').date()
            except ValueError:
                raise ValueError(
                    "Неверный формат даты. Используйте формат: YYYY-MM-DD (например: 2023-01-15)"
                )
        
        # Проверяем, что это объект date
        if not isinstance(v, date):
            raise ValueError("Поле 'date' должно быть датой")
        
        # Не разрешаем будущие даты
        today = date.today()
        if v > today:
            raise ValueError(
                f"Дата транзакции не может быть в будущем. Получено: {v.strftime('%Y-%m-%d')}, "
                f"сегодня: {today.strftime('%Y-%m-%d')}"
            )
        
        # Не разрешаем слишком старые даты (старше 10 лет)
        min_date = today - timedelta(days=3650)  # 10 лет назад
        if v < min_date:
            raise ValueError(
                f"Дата транзакции слишком старая. Минимальная дата: {min_date.strftime('%Y-%m-%d')}, "
                f"получено: {v.strftime('%Y-%m-%d')}"
            )
        
        return v


def validate_numeric_field(value: Optional[Union[str, float, int]], field_name: str, allow_negative: bool = False) -> Optional[float]:
    """
    Валидатор для числовых полей, который преобразует строки в числа
    
    Args:
        value: Значение поля (может быть строкой, числом или None)
        field_name: Название поля для сообщения об ошибке
        allow_negative: Разрешить отрицательные значения (по умолчанию False)
        
    Returns:
        float или None
        
    Raises:
        ValueError: Если значение не может быть преобразовано в число или отрицательное
    """
    if value is None:
        return None
    
    # Если уже число, возвращаем как float
    if isinstance(value, (int, float)):
        numeric_value = float(value)
    # Если строка, пытаемся преобразовать
    elif isinstance(value, str):
        # Убираем пробелы
        value = value.strip()
        
        # Пустая строка = None
        if value == '' or value.lower() in ('null', 'none', 'n/a', '-'):
            return None
        
        try:
            # Пробуем преобразовать в float
            numeric_value = float(value)
        except ValueError:
            raise ValueError(
                f"Поле '{field_name}' должно быть числом. Получено: '{value}'. "
                f"Пожалуйста, укажите числовое значение (например: 1000.50) или оставьте поле пустым."
            )
    else:
        # Если другой тип, выбрасываем ошибку
        raise ValueError(
            f"Поле '{field_name}' должно быть числом. Получен тип: {type(value).__name__}, значение: '{value}'"
        )
    
    # Проверка на отрицательные значения
    if not allow_negative and numeric_value < 0:
        raise ValueError(
            f"Поле '{field_name}' не может быть отрицательным. Получено: {numeric_value}. "
            f"Пожалуйста, укажите положительное число или ноль."
        )
    
    return numeric_value


class CSVTransactionInput(BaseModel):
    """Схема для входных данных транзакции в формате CSV"""
    Date: str
    RefNo: Optional[str] = None
    Withdrawal: Optional[float] = Field(None, ge=0, description="Сумма снятия (не может быть отрицательной)")
    Deposit: Optional[float] = Field(None, ge=0, description="Сумма пополнения (не может быть отрицательной)")
    Balance: Optional[float] = Field(None, ge=0, description="Баланс (не может быть отрицательным)")
    
    @field_validator('Withdrawal', mode='before')
    @classmethod
    def validate_withdrawal(cls, v):
        """Валидация поля Withdrawal (не может быть отрицательным)"""
        return validate_numeric_field(v, 'Withdrawal', allow_negative=False)
    
    @field_validator('Deposit', mode='before')
    @classmethod
    def validate_deposit(cls, v):
        """Валидация поля Deposit (не может быть отрицательным)"""
        return validate_numeric_field(v, 'Deposit', allow_negative=False)
    
    @field_validator('Balance', mode='before')
    @classmethod
    def validate_balance(cls, v):
        """Валидация поля Balance (не может быть отрицательным)"""
        return validate_numeric_field(v, 'Balance', allow_negative=False)
    
    @field_validator('Date', mode='before')
    @classmethod
    def validate_date(cls, v):
        """Валидация поля Date в CSV формате"""
        if v is None:
            raise ValueError("Поле 'Date' обязательно для заполнения")
        
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Поле 'Date' не может быть пустым")
            
            # Пробуем разные форматы дат
            date_formats = [
                '%Y-%m-%d',      # 2023-01-15
                '%d.%m.%Y',      # 15.01.2023
                '%d/%m/%Y',      # 15/01/2023
                '%Y/%m/%d',      # 2023/01/15
                '%d-%m-%Y',      # 15-01-2023
            ]
            
            parsed_date = None
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(v, fmt).date()
                    break
                except ValueError:
                    continue
            
            if parsed_date is None:
                raise ValueError(
                    f"Неверный формат даты: '{v}'. "
                    f"Используйте один из форматов: YYYY-MM-DD, DD.MM.YYYY, DD/MM/YYYY, DD-MM-YYYY"
                )
            
            # Проверка на разумность даты (не слишком далеко в будущем, не слишком старое)
            today = date.today()
            
            # Разрешаем будущие даты до 1 года вперед (для прогнозов)
            max_future_date = today + timedelta(days=365)
            if parsed_date > max_future_date:
                raise ValueError(
                    f"Дата не может быть более чем на 1 год в будущем. "
                    f"Получено: {parsed_date.strftime('%Y-%m-%d')}, "
                    f"максимум: {max_future_date.strftime('%Y-%m-%d')}"
                )
            
            # Не разрешаем слишком старые даты (старше 10 лет)
            min_date = today - timedelta(days=3650)  # 10 лет назад
            if parsed_date < min_date:
                raise ValueError(
                    f"Дата слишком старая. Минимальная дата: {min_date.strftime('%Y-%m-%d')}, "
                    f"получено: {parsed_date.strftime('%Y-%m-%d')}"
                )
            
            # Возвращаем в стандартном формате YYYY-MM-DD
            return parsed_date.strftime('%Y-%m-%d')
        
        # Если не строка, преобразуем в строку
        return str(v)


class CSVTransactionPredictionResponse(BaseModel):
    """Схема для ответа с предсказанной категорией"""
    Date: str
    RefNo: Optional[str] = None
    Withdrawal: Optional[float] = None
    Deposit: Optional[float] = None
    Balance: Optional[float] = None
    Category: str
    Probability: float = Field(..., ge=0.0, le=1.0)
