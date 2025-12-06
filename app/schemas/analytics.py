# app/schemas/analytics.py
from pydantic import BaseModel
from typing import List

class BalancePoint(BaseModel):
    date: str
    amount: float