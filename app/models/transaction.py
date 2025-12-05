from sqlalchemy import Column, Integer, String, Numeric, Date, ForeignKey, Boolean, DateTime, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
from app.models.category import TransactionCategory


class Transaction(Base):
    """Модель транзакции"""
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    category = Column(SQLEnum(TransactionCategory), nullable=False)
    amount = Column(Numeric(10, 2), nullable=False)
    description = Column(String, nullable=True)
    date = Column(Date, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Связь с пользователем
    user = relationship("User", back_populates="transactions")
    
    @property
    def is_income(self) -> bool:
        """Определяет, является ли транзакция доходом"""
        income_categories = [
            TransactionCategory.SALARY,
            TransactionCategory.BONUS,
            TransactionCategory.INVESTMENT,
            TransactionCategory.GIFT_INCOME,
            TransactionCategory.OTHER_INCOME
        ]
        return self.category in income_categories


