import joblib
import pandas as pd
import numpy as np

# Загрузка модели
model = joblib.load('classifier_v2.pkl')
print("Модель загружена.")

# === Повторим извлечение признаков ===
def extract_features(df):
    df = df.sort_values("Date").reset_index(drop=True)
    df["is_withdrawal"] = (df["Withdrawal"] > 0).astype(int)
    df["is_deposit"] = (df["Deposit"] > 0).astype(int)
    df["amount"] = df["Withdrawal"] + df["Deposit"]
    df["net_flow"] = df["Deposit"] - df["Withdrawal"]
    df["balance_before"] = df["Balance"] + df["Withdrawal"] - df["Deposit"]

    df["day_of_month"] = df["Date"].dt.day
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_month_start"] = (df["day_of_month"] <= 10).astype(int)
    df["is_month_end"] = (df["day_of_month"] >= 24).astype(int)

    # Точные бизнес-правила
    df["is_salary_like"] = (df["Deposit"] == 34800).astype(int)
    df["is_rent_like"] = (
        (df["Withdrawal"] >= 3900) & 
        (df["Withdrawal"] <= 7500) & 
        (df["day_of_month"] <= 6)
    ).astype(int)

    # Days since last salary
    salary_dates = df[df["Deposit"] == 34800]["Date"].tolist()
    df["days_since_last_salary"] = np.nan
    for i, row in df.iterrows():
        past_salaries = [d for d in salary_dates if d <= row["Date"]]
        if past_salaries:
            last_salary = max(past_salaries)
            df.at[i, "days_since_last_salary"] = (row["Date"] - last_salary).days

    # Days since last transaction
    df["days_since_last_txn"] = df["Date"].diff().dt.days.fillna(0)

    return df

# === Загрузка данных ===
df = pd.read_csv("ci_data.csv", skiprows=2)
df.columns = ["Date1", "Category", "RefNo", "Date2", "Withdrawal", "Deposit", "Balance"]
df["Date"] = pd.to_datetime(df["Date2"], format="mixed", errors="coerce")
df = df[["Date", "Category", "Withdrawal", "Deposit", "Balance"]].copy()
df["Withdrawal"] = pd.to_numeric(df["Withdrawal"], errors="coerce").fillna(0)
df["Deposit"] = pd.to_numeric(df["Deposit"], errors="coerce").fillna(0)
df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce").fillna(method="ffill")

# === Фильтрация ===
valid_cats = ["Food", "Misc", "Rent", "Salary", "Shopping"]
df_filtered = df[df["Category"].isin(valid_cats)].copy().reset_index(drop=True)

# === Извлечение признаков ===
df_features = extract_features(df_filtered)

# === Подготовка X ===
feature_columns = [
    "is_withdrawal", "is_deposit", "amount", "net_flow", "balance_before",
    "day_of_month", "day_of_week", "is_month_start", "is_month_end",
    "is_salary_like", "is_rent_like", "days_since_last_salary", "days_since_last_txn",
    "Withdrawal", "Deposit"
]

X = df_features[feature_columns]
X = X.fillna(-1)  # Заполняем NaN

# === Предсказание ===
y_pred_proba = model.predict_proba(X)
y_pred = model.predict(X)

# === Применение fallback rules ===
def apply_enhanced_fallback_rules_with_proba(X, y_pred_proba, categories=model.classes_):
    y_pred_proba = y_pred_proba.copy()
    for i in range(len(X)):
        row = X.iloc[i]
        # Salary: Deposit == 34800 и дата 24–26
        if row["is_salary_like"] == 1 and row["day_of_month"] in [24, 25, 26]:
            idx = list(categories).index("Salary")
            y_pred_proba[i] = 0
            y_pred_proba[i][idx] = 1
        # Rent: Withdrawal 3900–7500 и дата ≤6
        elif row["is_rent_like"] == 1:
            idx = list(categories).index("Rent")
            y_pred_proba[i] = 0
            y_pred_proba[i][idx] = 1
        # Shopping: 150–3000 в первые 5 дней после зарплаты
        elif (
            row["Withdrawal"] >= 150 and
            row["Withdrawal"] <= 3000 and
            row["days_since_last_salary"] >= 0 and
            row["days_since_last_salary"] <= 5 and
            row["is_withdrawal"] == 1
        ):
            if "Shopping" in categories:
                idx = list(categories).index("Shopping")
                y_pred_proba[i] = 0
                y_pred_proba[i][idx] = 1
    pred_indices = np.argmax(y_pred_proba, axis=1)
    pred_categories = categories[pred_indices]
    max_probas = np.max(y_pred_proba, axis=1)
    return pred_categories, max_probas

# Получаем предсказания и соответствующие вероятности
y_pred_hybrid, y_proba_hybrid = apply_enhanced_fallback_rules_with_proba(X, y_pred_proba)

# === Вывод результата с вероятностями (первые 100) ===
print("Предсказания и вероятности (первые 100):")
for i in range(min(100, len(y_pred_hybrid))):
    print(f"{y_pred_hybrid[i]} ({y_proba_hybrid[i]:.3f})")

print("\nИстинные категории (первые 100):")
print(df_features["Category"].values[:100])