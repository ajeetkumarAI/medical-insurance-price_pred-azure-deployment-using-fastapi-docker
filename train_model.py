import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Load data ──
df = pd.read_csv("medical_insurance.csv")
print(f"Dataset shape: {df.shape}")

# ── Encode categorical: sex & smoker ──
df["sex"] = df["sex"].replace({"female": 0, "male": 1})
df["smoker"] = df["smoker"].replace({"no": 0, "yes": 1})

# ── Handle BMI outliers (IQR capping) ──
q1 = df["bmi"].quantile(0.25)
q3 = df["bmi"].quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr
df["bmi"] = np.where(df["bmi"] > upper_limit, upper_limit, df["bmi"])

# ── One-hot encode region ──
df = pd.get_dummies(df, columns=["region"], dtype=int)

# ── Split features and target ──
x = df.drop("charges", axis=1)
y = df["charges"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=2
)

# ── Train model ──
model = LinearRegression()
model.fit(x_train, y_train)


# ── Evaluate ──
def evaluate(label, model, x_data, y_data):
    pred = model.predict(x_data)
    mse = mean_squared_error(y_data, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_data, pred)
    r2 = r2_score(y_data, pred)
    adj_r2 = 1 - (((1 - r2) * (x_data.shape[0] - 1)) / (x_data.shape[0] - x_data.shape[1] - 1))
    print(f"\n{'─' * 40}")
    print(f"  {label}")
    print(f"{'─' * 40}")
    print(f"  MSE      : {mse:.2f}")
    print(f"  RMSE     : {rmse:.2f}")
    print(f"  MAE      : {mae:.2f}")
    print(f"  R2       : {r2:.4f}")
    print(f"  Adj-R2   : {adj_r2:.4f}")


evaluate("Train Data", model, x_train, y_train)
evaluate("Test Data", model, x_test, y_test)

# ── Save model as pickle ──
with open("Linear_Model.pkl", "wb") as f:
    pickle.dump(model, f)
print("\nModel saved: Linear_Model.pkl")

# ── Save project metadata as JSON ──
project_data = {
    "sex": {"female": 0, "male": 1},
    "smoker": {"no": 0, "yes": 1},
    "columns": list(x_train.columns),
}
with open("project_data.json", "w") as f:
    json.dump(project_data, f)
print("Metadata saved: project_data.json")
