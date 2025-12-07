"""API endpoints для дообучения модели"""
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any

from app.database import get_db
from app.models.user import User
from app.dependencies import get_current_user
from app.services.model_retraining import retrain_model

router = APIRouter(prefix="/ml", tags=["machine-learning"])


@router.post("/retrain", response_model=Dict[str, Any])
async def retrain_model_endpoint(
    days_back: int = Query(7, ge=1, le=365, description="Количество дней назад для выборки новых транзакций"),
    original_csv_path: Optional[str] = Query(None, description="Путь к оригинальному CSV файлу (опционально)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Ручной запуск дообучения модели на основе новых данных пользователей
    
    Требует авторизации. Дообучает модель на транзакциях всех пользователей
    за указанный период (по умолчанию 7 дней).
    
    Args:
        days_back: Количество дней назад для выборки транзакций (1-365)
        original_csv_path: Путь к оригинальному CSV файлу для объединения данных
        current_user: Текущий авторизованный пользователь
        db: Сессия базы данных
        
    Returns:
        Результат дообучения с метриками
    """
    try:
        result = retrain_model(
            db=db,
            original_csv_path=original_csv_path,
            days_back=days_back
        )
        
        if result["success"]:
            return {
                "success": True,
                "message": result["message"],
                "new_transactions_count": result["new_transactions_count"],
                "metrics": result.get("metrics", {}),
                "model_path": result.get("model_path", "")
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при дообучении модели: {str(e)}"
        )




