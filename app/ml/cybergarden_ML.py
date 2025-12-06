import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === 1. Загрузка и очистка данных ===
df = pd.read_csv("ci_data.csv", skiprows=5)
df.columns = ["Date1", "Category", "RefNo", "Date2", "Withdrawal", "Deposit", "Balance"]
df["Date"] = pd.to_datetime(df["Date2"], format="mixed", errors="coerce")
df = df[["Date", "Category", "Withdrawal", "Deposit", "Balance"]].copy()
df["Withdrawal"] = pd.to_numeric(df["Withdrawal"], errors="coerce").fillna(0)
df["Deposit"] = pd.to_numeric(df["Deposit"], errors="coerce").fillna(0)
df["Balance"] = pd.to_numeric(df["Balance"], errors="coerce").fillna(method="ffill")

# === 2. Фильтрация категорий и очистка опечаток ===
# Заменяем редкую категорию Transport на Misc
df["Category"] = df["Category"].replace("Transport", "Misc")

# Теперь фильтруем только устойчивые категории
valid_cats = ["Food", "Misc", "Rent", "Salary", "Shopping"]
df = df[df["Category"].isin(valid_cats)].copy().reset_index(drop=True)

# === 3. Извлечение признаков с учётом контекста ===
def extract_features(df):
    df = df.sort_values("Date").reset_index(drop=True)
    df["is_withdrawal"] = (df["Withdrawal"] > 0).astype(int)
    df["is_deposit"] = (df["Deposit"] > 0).astype(int)
    df["amount"] = df["Withdrawal"] + df["Deposit"]
    df["net_flow"] = df["Deposit"] - df["Withdrawal"]
    df["balance_before"] = df["Balance"] + df["Withdrawal"] - df["Deposit"]  # баланс ДО транзакции

    df["day_of_month"] = df["Date"].dt.day
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_month_start"] = (df["day_of_month"] <= 10).astype(int)    # расширено до 10
    df["is_month_end"] = (df["day_of_month"] >= 24).astype(int)     # 24–31

    # Признаки на основе бизнес-логики
    df["is_salary_like"] = (df["Deposit"] == 34800).astype(int)     # точное совпадение
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

    # Days since last transaction (активность)
    df["days_since_last_txn"] = df["Date"].diff().dt.days.fillna(0)

    return df

df_features = extract_features(df)

# === 4. Подготовка X и y ===
feature_columns = [
    "is_withdrawal", "is_deposit", "amount", "net_flow", "balance_before",
    "day_of_month", "day_of_week", "is_month_start", "is_month_end",
    "is_salary_like", "is_rent_like", "days_since_last_salary", "days_since_last_txn",
    "Withdrawal", "Deposit"  # ← добавлено
]

X = df_features[feature_columns].fillna(-1)
y = df_features["Category"]

# === 5. Разделение по времени: train до июля, test — июль–декабрь ===
split_date = "2023-07-01"
train_idx = df_features[df_features["Date"] < split_date].index
test_idx = df_features[df_features["Date"] >= split_date].index

X_train = X.loc[train_idx]
y_train = y.loc[train_idx]
X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print("Support in test:", y_test.value_counts().sort_index())

# === 6. Обучение модели с регуляризацией ===
model = lgb.LGBMClassifier(
    n_estimators=100,
    num_leaves=15,
    learning_rate=0.05,
    min_data_in_leaf=10,
    lambda_l1=0.1,
    lambda_l2=0.1,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# === 7. Применение fallback-правил (более надёжных) ===
def apply_fallback_rules(X, y_pred, categories):
    y_pred = y_pred.copy()
    for i in range(len(X)):
        row = X.iloc[i]
        # Rule 1: Salary
        if row["is_salary_like"] == 1 and row["day_of_month"] in [24, 25, 26]:
            y_pred[i] = "Salary"
        # Rule 2: Rent
        elif row["is_rent_like"] == 1:
            y_pred[i] = "Rent"
        # Rule 3: Shopping — крупная трата (150–3000) в первые 5 дней после зарплаты
        elif (
            row["Withdrawal"] >= 150 and
            row["Withdrawal"] <= 3000 and
            row["days_since_last_salary"] >= 0 and
            row["days_since_last_salary"] <= 5 and
            row["is_withdrawal"] == 1
        ):
            y_pred[i] = "Shopping"
    return y_pred

y_pred = model.predict(X_test)
y_pred_hybrid = apply_fallback_rules(X_test, y_pred, model.classes_)

# === 8. Оценка ===
print("\n=== Оценка модели ===")
print(classification_report(y_test, y_pred_hybrid, zero_division=0))
print("F1 (weighted):", f1_score(y_test, y_pred_hybrid, average="weighted"))

# === 9. Confusion Matrix ===
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_hybrid, labels=valid_cats)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=valid_cats, yticklabels=valid_cats, cmap='Blues')
plt.title("Confusion Matrix (Hybrid Model)")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix_v2.png", dpi=150)

# === 12. Расширенный анализ качества модели ===

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Общая точность и F1
overall_f1 = f1_score(y_test, y_pred_hybrid, average="weighted")
print(f"Общий F1-score (weighted): {overall_f1:.4f}")

# Оценим, сколько из этих случаев модель бы ошиблась без правил
# salary_rule_cases = test_with_rules[
#     (test_with_rules["is_deposit"] == 1) &
#     (test_with_rules["amount"] > 30000) &
#     (test_with_rules["day_of_month"].isin([24, 25, 26]))
# ]
# rent_rule_cases = test_with_rules[
#     (test_with_rules["is_withdrawal"] == 1) &
#     (test_with_rules["amount"] > 5000) &
#     (test_with_rules["day_of_month"] <= 6)
# ]

# salary_fixed = (salary_rule_cases["predicted_before_fallback"] != "Salary").sum()
# rent_fixed = (rent_rule_cases["predicted_before_fallback"] != "Rent").sum()

# print(f"Без правил модель ошиблась бы в {salary_fixed} из {len(salary_rule_cases)} Salary-транзакций")
# print(f"Без правил модель ошиблась бы в {rent_fixed} из {len(rent_rule_cases)} Rent-транзакций")

# 4. Матрица ошибок
print("\n Матрица ошибок: confusion_matrix.png \n")
cm = confusion_matrix(y_test, y_pred_hybrid, labels=valid_cats)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=valid_cats, yticklabels=valid_cats, cmap='Blues')
plt.title("Confusion Matrix (Hybrid Model)")
plt.ylabel("Истинная категория")
plt.xlabel("Предсказанная категория")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)


# 6. Проверка на переобучение (сравнение train/test F1)
y_train_pred = model.predict(X_train)
train_f1 = f1_score(y_train, y_train_pred, average="weighted")
print(f"Признаки переобучения:")
print(f"F1 на train: {train_f1:.4f}")
print(f"F1 на test:  {overall_f1:.4f}")

# 7. Рекомендации
# if report_dict.get("Misc", {}).get('f1-score', 0) < 0.6:
#     print("Категория 'Misc' имеет низкое качество → рассмотрите добавление более специфичных признаков или подкатегорий.")
# if fallback_salary_hits == 0 or fallback_rent_hits == 0:
#     print("Fallback-правила не сработали → проверьте, соответствуют ли данные ожидаемому поведению (например, зарплата 25 числа).")
# if overall_f1 < 0.85:
#     print("Общий F1 < 0.85 → рассмотрите расширение признаков (например, скользящие средние, месячные агрегаты).")
# else:
#     print("качество (F1 ≥ 0.85).")

# print("Обучение завершено. Модель готова к использованию.")

# === 10. Сохранение модели ===
joblib.dump(model, "classifier_v2.pkl")
print("\n✅ Модель сохранена как 'classifier_v2.pkl'")