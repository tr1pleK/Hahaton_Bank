# Авторизация в Finance Analysis API

## Демонстрационный пользователь

При первом запуске приложения автоматически создается демонстрационный пользователь:

- **Email:** `demo@finance.app`
- **Пароль:** `demo123`
- **Имя:** Демо Пользователь
- **Баланс:** 10000.00
- **Тариф:** free (бесплатный)

## API Endpoints

### 1. Вход в систему
```http
POST /auth/login
Content-Type: application/json

{
  "email": "demo@finance.app",
  "password": "demo123"
}
```

**Ответ:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### 2. Получить данные текущего пользователя
```http
GET /auth/me
Authorization: Bearer <access_token>
```

**Ответ:**
```json
{
  "id": 1,
  "email": "demo@finance.app",
  "full_name": "Демо Пользователь",
  "balance": 10000.00
}
```

## Использование токена

Для доступа к защищенным endpoints добавьте заголовок:
```
Authorization: Bearer <ваш_токен>
```

## Создание демо-пользователя вручную

Если нужно создать демо-пользователя вручную:

```bash
py scripts/create_demo_user.py
```

## Использование в коде

Для защиты endpoint используйте dependency:

```python
from app.dependencies import get_current_user
from app.models.user import User

@router.get("/protected")
async def protected_route(current_user: User = Depends(get_current_user)):
    return {"message": f"Привет, {current_user.full_name}!"}
```

