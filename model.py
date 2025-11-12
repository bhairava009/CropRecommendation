# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import os

CSV_PATH = os.path.join('data', 'Crop_recommendation.csv')
MODEL_PATH = 'crop_model.pkl'

def load_data(path):
    df = pd.read_csv(path)
    print("✅ Loaded dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

def prepare(df):
    # Expecting a target column named 'label'
    if 'label' not in df.columns:
        raise ValueError("Expected target column named 'label' in CSV. Found: " + ", ".join(df.columns))
    X = df.drop('label', axis=1)
    y = df['label']
    return X, y

def train_and_save(X, y, model_path=MODEL_PATH):
    # Split before SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

    # 🔹 Apply SMOTE to handle class imbalance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("After SMOTE:", X_train_res.shape, y_train_res.shape)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc*100:.2f}%")
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    joblib.dump(model, model_path)
    print(f"💾 Saved model to {model_path}")
    return acc

if __name__ == "__main__":
    df = load_data(CSV_PATH)
    X, y = prepare(df)
    train_and_save(X, y)
