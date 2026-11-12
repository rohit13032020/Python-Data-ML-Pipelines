
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer


df = pd.read_csv("student-data.csv")


# 1. Clean data
def clean_data(df):
    df = df.fillna(0)
    return df


# 2. Remove duplicates
def remove_duplicates(df):
    return df.drop_duplicates()


# 3. Split data
def split_data(df, target_columns):
    X = df.drop(columns=[target_columns])
    y = df[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# 4. One Hot Encoding function
def one_hot_encoding():
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )
    return encoder


# 5. Create ML Pipeline
def create_pipeline(X_train):

    numerical_columns = X_train.select_dtypes(
        include=["number"]
    ).columns

    categorical_columns = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numerical_columns),
        ("cat", one_hot_encoding(), categorical_columns)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    

    return pipeline


# 6. Train model
def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


# 7. Evaluate model
def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", accuracy)


# Main Pipeline Execution

df = clean_data(df)
df = remove_duplicates(df)

X_train, X_test, y_train, y_test = split_data(
    df, target_columns="passed"
)

pipeline = create_pipeline(X_train)

pipeline = train_model(pipeline, X_train, y_train)

evaluate_model(pipeline, X_test, y_test)
