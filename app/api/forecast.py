from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.services.balance_forecast_service import get_balance_forecast

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.get("/balance")
async def get_balance_forecast_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Получение прогноза баланса на конец текущего месяца для текущего пользователя.
    
    Returns:
        Словарь с прогнозом баланса, ежедневным прогнозом и рекомендациями
    """
    try:
        result = get_balance_forecast(db, current_user.id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при получении прогноза: {str(e)}")


