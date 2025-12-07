"""API endpoints для транзакций"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from datetime import date
import pandas as pd
from pydantic import ValidationError

from app.database import get_db
from app.models.user import User
from app.models.transaction import Transaction
from app.models.category import TransactionCategory
from app.dependencies import get_current_user
from app.schemas.transaction import (
    TransactionCreate, TransactionUpdate,
    CSVTransactionInput, CSVTransactionPredictionResponse,
    TransactionFrontendDto, TransactionPredictDto, TransactionDataBaseDto,
    TransactionCategoryUpdate
)
from app.services.transaction_service import (
    create_transaction, get_transaction, get_transactions,
    update_transaction, delete_transaction
)
from app.ml.transaction_classifier import TransactionClassifier

router = APIRouter(prefix="/transactions", tags=["transactions"])


def to_dict(transaction) -> Dict[str, Any]:
    """Преобразует транзакцию в словарь"""
    cat = transaction.category
    return {
        "id": transaction.id,
        "user_id": transaction.user_id,
        "amount": float(transaction.amount),
        "description": transaction.description,
        "date": transaction.date.isoformat() if transaction.date else None,
        "category": cat.value if hasattr(cat, 'value') else str(cat),
        "is_income": transaction.is_income,
        "created_at": transaction.created_at.isoformat() if transaction.created_at else None
    }


@router.post("", status_code=status.HTTP_201_CREATED, response_model=TransactionDataBaseDto)
async def create_transaction_endpoint(
    transaction_data: TransactionFrontendDto,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> TransactionDataBaseDto:
    """
    Создать транзакцию с автоматическим определением категории через ML модель.
    
    Принимает данные с фронтенда (date, isIncome, value) и:
    1. Преобразует в формат для ML модели
 fix_trans
    2. Получает предсказание категории (только для расходов, для доходов используется дефолтная категория)
    3. Сохраняет транзакцию в БД
    4. Возвращает полную информацию о транзакции с категорией
    
    Важно: Если isIncome = true, категория не предсказывается, используется OTHER_INCOME.
    Поле isIncome в ответе всегда соответствует значению, указанному пользователем.

    2. Получает предсказание категории
    3. Сохраняет транзакцию в БД
    4. Возвращает полную информацию о транзакции с категорией
 main
    """
    try:
        # 1. Парсим дату
        from datetime import datetime as dt
        transaction_date = dt.strptime(transaction_data.date, '%Y-%m-%d').date()
        
        # 2. Получаем текущий баланс пользователя
        db.refresh(current_user)
        current_balance = float(current_user.balance)
        
        # 3. Рассчитываем баланс после транзакции
        if transaction_data.isIncome:
            new_balance = current_balance + transaction_data.value
            withdrawal = 0.0
            deposit = transaction_data.value
        else:
            new_balance = current_balance - transaction_data.value
            withdrawal = transaction_data.value
            deposit = 0.0
        
        # 4. Формируем TransactionPredictDto для модели
        predict_dto = TransactionPredictDto(
            Date=transaction_data.date,
            date=transaction_data.date,
            Ref_num='',
            Withdrawal=withdrawal,
            Deposit=deposit,
            Balance=new_balance
        )
        
        # 5. Создаем DataFrame для модели
        df = pd.DataFrame([{
            'Date': transaction_date,
            'RefNo': predict_dto.Ref_num or '',
            'Withdrawal': predict_dto.Withdrawal,
            'Deposit': predict_dto.Deposit,
            'Balance': predict_dto.Balance
        }])
        
        # 6. Получаем предсказание категории от ML модели
 fix_trans
        # Если isIncome = true, не предсказываем категорию, используем дефолтную категорию дохода
        # Если isIncome = false, предсказываем категорию как обычно
        if transaction_data.isIncome:
            # Для доходов не предсказываем категорию, используем дефолтную
            predicted_category = TransactionCategory.OTHER_INCOME
            probability = 1.0  # 100% уверенность, так как пользователь явно указал доход
        else:
            # Для расходов предсказываем категорию через ML модель
            classifier = TransactionClassifier()
            if not classifier.is_trained:
                # Если модель не загружена, используем дефолтную категорию расхода
                predicted_category = TransactionCategory.OTHER_EXPENSE
                probability = 0.5
            else:
                predicted_category, probability = classifier.predict_from_dataframe(df)

        classifier = TransactionClassifier()
        if not classifier.is_trained:
            # Если модель не загружена, используем дефолтную категорию
            predicted_category = TransactionCategory.OTHER_EXPENSE
            probability = 0.5
        else:
            predicted_category, probability = classifier.predict_from_dataframe(df)
 main
        
        # 7. Создаем транзакцию в БД
        transaction = Transaction(
            user_id=current_user.id,
            amount=transaction_data.value,
            description=None,
            date=transaction_date,
            category=predicted_category
        )
        
        db.add(transaction)
        
        # 8. Обновляем баланс пользователя
        current_user.balance = new_balance
        
        db.commit()
        db.refresh(transaction)
        db.refresh(current_user)
        
        # 9. Формируем ответ TransactionDataBaseDto
 fix_trans
        # Гарантируем, что is_income соответствует значению, указанному пользователем
        # (transaction.is_income уже будет правильным, так как мы установили правильную категорию)

 main
        return TransactionDataBaseDto(
            id=transaction.id,
            user_id=transaction.user_id,
            date=transaction.date.isoformat(),
            amount=float(transaction.amount),
            description=transaction.description,
            category=predicted_category.value,
            category_probability=round(probability, 4),
 fix_trans
            is_income=transaction_data.isIncome,  # Используем значение, указанное пользователем

            is_income=transaction.is_income,
 main
            created_at=transaction.created_at.isoformat() if transaction.created_at else None
        )
        
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при создании транзакции: {str(e)}"
        )


@router.get("")
async def get_transactions_endpoint(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    category: Optional[str] = Query(None),
    is_income: Optional[bool] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Получить список транзакций"""
    # Валидация диапазона дат
    if start_date and end_date:
        if start_date > end_date:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"start_date не может быть позже end_date. "
                    f"Получено: start_date={start_date}, end_date={end_date}"
                )
            )
        
        # Проверка на слишком большой диапазон (больше 5 лет)
        delta = end_date - start_date
        max_days = 1825  # 5 лет
        if delta.days > max_days:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Диапазон дат слишком большой. Максимум: {max_days // 365} лет "
                    f"({max_days} дней). Получено: {delta.days} дней"
                )
            )
    
    # Валидация отдельных дат
    today = date.today()
    if start_date and start_date > today:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"start_date не может быть в будущем. Получено: {start_date}, сегодня: {today}"
        )
    
    if end_date and end_date > today:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"end_date не может быть в будущем. Получено: {end_date}, сегодня: {today}"
        )
    
    category_enum = None
    if category:
        try:
            category_enum = TransactionCategory(category)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Неверная категория: {category}")
    
    transactions, total = get_transactions(
        db, current_user.id, skip, limit, start_date, end_date, category_enum, is_income
    )
    
    return {
        "transactions": [to_dict(t) for t in transactions],
        "total": total,
        "page": skip // limit + 1 if limit > 0 else 1,
        "page_size": limit
    }


@router.get("/{transaction_id}")
async def get_transaction_endpoint(
    transaction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Получить транзакцию по ID"""
    transaction = get_transaction(db, transaction_id, current_user.id)
    if not transaction:
        raise HTTPException(status_code=404, detail="Транзакция не найдена")
    return to_dict(transaction)


@router.put("/{transaction_id}")
async def update_transaction_endpoint(
    transaction_id: int,
    transaction_data: TransactionCategoryUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Обновить категорию транзакции"""
    try:
        # Получаем транзакцию
        transaction = get_transaction(db, transaction_id, current_user.id)
        if not transaction:
            raise HTTPException(status_code=404, detail="Транзакция не найдена")
        
        # Валидируем и обновляем категорию
        try:
            category = TransactionCategory(transaction_data.category)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Неверная категория: {transaction_data.category}"
            )
        
        transaction.category = category
        db.commit()
        db.refresh(transaction)
        
        return to_dict(transaction)
    except HTTPException:
        raise
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обновлении транзакции: {str(e)}"
        )


@router.delete("/{transaction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_transaction_endpoint(
    transaction_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Удалить транзакцию"""
    if not delete_transaction(db, transaction_id, current_user.id):
        raise HTTPException(status_code=404, detail="Транзакция не найдена")
    return None


@router.post("/predict-category", response_model=List[CSVTransactionPredictionResponse])
async def predict_category_endpoint(
    transactions: List[CSVTransactionInput],
    current_user: User = Depends(get_current_user)
) -> List[CSVTransactionPredictionResponse]:
    """
    Предсказание категорий для массива транзакций с использованием ML модели
    
    Принимает массив транзакций в формате CSV (Date, RefNo, Withdrawal, Deposit, Balance)
    и возвращает массив с предсказанными категориями и вероятностями.
    
    Args:
        transactions: Массив транзакций для классификации (минимум 1 элемент)
        
    Returns:
        Массив транзакций с добавленными полями Category и Probability
    """
    try:
        if not transactions or len(transactions) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Массив транзакций не может быть пустым"
            )
        
        # Инициализация классификатора
        classifier = TransactionClassifier()
        
        if not classifier.is_trained:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Модель не загружена. Убедитесь, что файл classifier_v2.pkl существует в app/ml/"
            )
        
        # Результаты предсказаний
        results = []
        
        # Обрабатываем каждую транзакцию
        for transaction in transactions:
            # Преобразование данных в DataFrame (как в test_classifier.py)
            df = pd.DataFrame([{
                'Date': transaction.Date,
                'RefNo': transaction.RefNo or '',
                'Withdrawal': transaction.Withdrawal if transaction.Withdrawal else 0.0,
                'Deposit': transaction.Deposit if transaction.Deposit else 0.0,
                'Balance': transaction.Balance if transaction.Balance else 0.0
            }])
            
            # Применение логики из test_classifier.py
            category, probability = classifier.predict_from_dataframe(df)
            
            # Формирование ответа
            result = CSVTransactionPredictionResponse(
                Date=transaction.Date,
                RefNo=transaction.RefNo,
                Withdrawal=transaction.Withdrawal,
                Deposit=transaction.Deposit,
                Balance=transaction.Balance,
                Category=category.value,
                Probability=round(probability, 4)
            )
            results.append(result)
        
        return results
    except HTTPException:
        raise
    except ValidationError as e:
        # Обработка ошибок валидации Pydantic
        errors = []
        for error in e.errors():
            field = " -> ".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            errors.append({
                "field": field,
                "message": message,
                "type": error.get("type", "validation_error")
            })
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Ошибка валидации данных",
                "errors": errors,
                "message": "Проверьте правильность введенных данных. Все числовые поля должны содержать числа."
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при предсказании категории: {str(e)}"
        )
