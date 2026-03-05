# =====================================================
# 1. Import Required Libraries
# =====================================================

import numpy as np
import pandas as pd
import pickle

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# =====================================================
# 2. Load Dataset
# =====================================================

# Load housing dataset
df = pd.read_csv("/Users/akshatthakur22/Desktop/python/project_1/dataset/Housing.csv")

print("\nDataset Loaded Successfully\n")


# =====================================================
# 3. Check Missing Values
# =====================================================

print("Missing Values in Dataset:\n")
print(df.isnull().sum())


# =====================================================
# 4. Handle Missing Values
# =====================================================

# Fill missing values with appropriate statistics

df["price"].fillna(df["price"].mean(), inplace=True)
df["bedrooms"].fillna(df["bedrooms"].mode()[0], inplace=True)
df["bathrooms"].fillna(df["bathrooms"].mode()[0], inplace=True)
df["stories"].fillna(df["stories"].mode()[0], inplace=True)


# =====================================================
# 5. Encode Categorical Variables
# =====================================================

# Columns containing categorical data
categorical_cols = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]

encoder = LabelEncoder()

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])


# Check dataset info
print("\nDataset Information:\n")
print(df.info())


# =====================================================
# 6. Define Features (X) and Target (y)
# =====================================================

X = df.drop("price", axis=1)   # input features
y = df["price"]                # target variable


# =====================================================
# 7. Split Dataset (Train / Test)
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# 8. Create Machine Learning Pipeline
# =====================================================

# Pipeline = Scaling + Model

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])


# =====================================================
# 9. Train the Model
# =====================================================

pipeline.fit(X_train, y_train)


# =====================================================
# 10. Make Predictions
# =====================================================

y_pred = pipeline.predict(X_test)


# =====================================================
# 11. Evaluate Model Performance
# =====================================================

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Performance")
print("-------------------------")
print("R² Score :", r2)
print("MSE      :", mse)

# =====================================================
# 12. Save the Model Pipeline
# =====================================================



with open("house_price_pipeline.pkl", "wb") as file:
    pickle.dump(pipeline, file)

print("\nPipeline model saved successfully!")