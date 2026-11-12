
# ==========================================
# LESSON 5: COMPLETE MACHINE LEARNING PIPELINE
# ==========================================

# 1. Import Libraries

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# 2. Create Dataset

data = {
    "Age": [20, 25, 30, 35, 40, 45, 28, 32, 50, 38],
    "Salary": [20000, 30000, 40000, 50000, 60000,
               70000, 35000, 45000, 80000, 55000],
    "City": ["Kolkata", "Delhi", "Mumbai", "Kolkata",
             "Delhi", "Mumbai", "Kolkata", "Delhi",
             "Mumbai", "Kolkata"],
    "Purchased": [0, 0, 1, 1, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

print(df.head())


# 3. Separate Features and Target

X = df.drop("Purchased", axis=1)
y = df["Purchased"]


# 4. Identify Numerical and Categorical Columns

numerical_cols = ["Age", "Salary"]
categorical_cols = ["City"]


# 5. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# 6. Create Numerical Pipeline

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# 7. Create Categorical Pipeline

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# 8. Create ColumnTransformer

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])


# 9. Create Complete ML Pipeline

model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])


# 10. Train Complete Pipeline

model_pipeline.fit(X_train, y_train)


# 11. Make Predictions

y_pred = model_pipeline.predict(X_test)

print("Predictions:", y_pred)


# 12. Evaluate Model

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# 13. Save Complete Pipeline

joblib.dump(model_pipeline, "ml_pipeline.pkl")

print("Pipeline saved successfully!")


# 14. Load Complete Pipeline

loaded_pipeline = joblib.load("ml_pipeline.pkl")


# 15. Predict on New Data

new_data = pd.DataFrame({
    "Age": [29],
    "Salary": [45000],
    "City": ["Kolkata"]
})

prediction = loaded_pipeline.predict(new_data)

print("New Prediction:", prediction)
