# train_model.py
import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


DATA_PATH = "data/train.csv"
MODELS_DIR = "models"
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")


def clean_text(s: str) -> str:
    
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)          # ссылки
    s = re.sub(r"[^a-zа-яё0-9\s]+", " ", s)          # всё кроме букв/цифр/пробелов
    s = re.sub(r"\s+", " ", s).strip()               # лишние пробелы
    return s


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    
    df = pd.read_csv(DATA_PATH, sep=";")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("В train.csv должны быть колонки: text и label")

    df["text"] = df["text"].apply(clean_text)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)


    df = df[df["text"].str.len() > 0].copy()

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    
    vectorizer = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        lowercase=False, 
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000, n_jobs=None)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)

    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print(classification_report(y_test, preds, digits=4))

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    print(f"\n✅ Сохранено:\n- {VECTORIZER_PATH}\n- {MODEL_PATH}")


if __name__ == "__main__":
    main()
