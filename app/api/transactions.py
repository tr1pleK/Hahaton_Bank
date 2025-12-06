"""API endpoints для транзакций"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any, List
from datetime import date
import pandas as pd

from app.database import get_db
from app.models.user import User
from app.models.category import TransactionCategory
from app.dependencies import get_current_user
from app.schemas.transaction import (
    TransactionCreate, TransactionUpdate,
    CSVTransactionInput, CSVTransactionPredictionResponse
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


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_transaction_endpoint(
    transaction_data: TransactionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Создать транзакцию"""
    transaction = create_transaction(db, current_user.id, transaction_data)
    return to_dict(transaction)


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
    transaction_data: TransactionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Обновить транзакцию"""
    try:
        transaction = update_transaction(db, transaction_id, current_user.id, transaction_data)
        if not transaction:
            raise HTTPException(status_code=404, detail="Транзакция не найдена")
        return to_dict(transaction)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при предсказании категории: {str(e)}"
        )
