from sqlalchemy import Column, Integer, String, Numeric, Date, ForeignKey, Boolean, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class Billing(Base):
    """Модель тарифного плана"""
    __tablename__ = "billings"
    
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, nullable=False, unique=True)  # "free", "premium", "pro"
    price = Column(Numeric(10, 2), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Связь с пользователями через UserBilling
    user_billings = relationship("UserBilling", back_populates="billing")


class UserBilling(Base):
    """Связь пользователя с тарифным планом"""
    __tablename__ = "user_billings"
    
    user_id = Column(Integer, ForeignKey("users.id"), primary_key=True)
    billing_id = Column(Integer, ForeignKey("billings.id"), primary_key=True)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=True)  # null = бессрочная подписка
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Связи
    user = relationship("User", back_populates="user_billings")
    billing = relationship("Billing", back_populates="user_billings")


