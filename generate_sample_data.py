import os
import pandas as pd
import numpy as np

os.makedirs('data', exist_ok=True)
np.random.seed(42)

# 1. Employee Attrition (600 rows)
n = 600
ages = np.random.normal(37, 8, n).clip(21, 62).astype(int)
job_roles = np.random.choice(
    ['Sales Executive', 'Research Scientist', 'Software Engineer', 'Manager', 'HR Specialist'],
    size=n, p=[0.3, 0.25, 0.25, 0.1, 0.1]
)
base_income = {
    'Sales Executive': 6500,
    'Research Scientist': 5800,
    'Software Engineer': 7200,
    'Manager': 12500,
    'HR Specialist': 5000
}
incomes = [int(base_income[role] + (age - 25) * 180 + np.random.normal(0, 600)) for role, age in zip(job_roles, ages)]
incomes = np.clip(incomes, 3000, 25000)
years_at_co = [int(min(age - 20, np.random.exponential(5))) for age in ages]
overtimes = np.random.choice(['Yes', 'No'], size=n, p=[0.35, 0.65])
work_life_balance = np.random.choice([1, 2, 3, 4], size=n, p=[0.1, 0.25, 0.5, 0.15])

# Attrition probability depends on overtime, lower income, lower work life balance
attrition = []
for ot, inc, wlb, yrs in zip(overtimes, incomes, work_life_balance, years_at_co):
    prob = 0.12
    if ot == 'Yes': prob += 0.18
    if inc < 5500: prob += 0.15
    if wlb == 1: prob += 0.20
    if yrs < 2: prob += 0.10
    attrition.append('Yes' if np.random.rand() < min(0.85, prob) else 'No')

df_employee = pd.DataFrame({
    'Age': ages,
    'JobRole': job_roles,
    'MonthlyIncome': incomes,
    'YearsAtCompany': years_at_co,
    'WorkLifeBalance': work_life_balance,
    'OverTime': overtimes,
    'Attrition': attrition
})
df_employee.to_csv('data/employee_attrition.csv', index=False)

# 2. Customer Churn (500 rows)
n_c = 500
tenure = np.random.randint(1, 72, size=n_c)
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_c, p=[0.55, 0.25, 0.20])
monthly_charges = np.random.uniform(20.0, 115.0, size=n_c).round(2)
total_charges = (monthly_charges * tenure + np.random.normal(0, 15, size=n_c)).round(2)
total_charges = np.maximum(total_charges, monthly_charges)
senior = np.random.choice([0, 1], size=n_c, p=[0.82, 0.18])
payment = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], size=n_c)

churn = []
for c, mc, ten, pay in zip(contract, monthly_charges, tenure, payment):
    p = 0.15
    if c == 'Month-to-month': p += 0.25
    if mc > 80: p += 0.15
    if ten < 12: p += 0.15
    if pay == 'Electronic check': p += 0.10
    churn.append('Yes' if np.random.rand() < min(0.9, p) else 'No')

df_churn = pd.DataFrame({
    'SeniorCitizen': senior,
    'Tenure': tenure,
    'Contract': contract,
    'PaymentMethod': payment,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Churn': churn
})
df_churn.to_csv('data/customer_churn.csv', index=False)

# 3. Medical Heart Disease (400 rows)
n_h = 400
h_age = np.random.normal(54, 9, n_h).clip(29, 77).astype(int)
h_sex = np.random.choice([1, 0], size=n_h, p=[0.68, 0.32]) # 1=male, 0=female
h_cp = np.random.choice([0, 1, 2, 3], size=n_h, p=[0.48, 0.17, 0.28, 0.07])
h_trestbps = np.random.normal(131, 17, n_h).clip(94, 200).astype(int)
h_chol = np.random.normal(246, 51, n_h).clip(126, 564).astype(int)
h_thalach = (220 - h_age - np.random.normal(15, 10, n_h)).clip(71, 202).astype(int)
h_exang = np.random.choice([0, 1], size=n_h, p=[0.67, 0.33])

h_target = []
for cp, th, ex, ch in zip(h_cp, h_thalach, h_exang, h_chol):
    p = 0.25
    if cp > 0: p += 0.35
    if th > 150: p += 0.20
    if ex == 0: p += 0.15
    if ch > 260: p += 0.05
    h_target.append(1 if np.random.rand() < min(0.92, p) else 0)

df_heart = pd.DataFrame({
    'age': h_age,
    'sex': h_sex,
    'cp': h_cp,
    'trestbps': h_trestbps,
    'chol': h_chol,
    'thalach': h_thalach,
    'exang': h_exang,
    'target': h_target
})
df_heart.to_csv('data/heart_disease.csv', index=False)
print("Demo datasets generated successfully in data/")
