import joblib


class TextVectorizer:
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    @staticmethod
    def load(path: str) -> "TextVectorizer":
        vec = joblib.load(path)
        if not hasattr(vec, "transform"):
            raise TypeError(f"Из {path} загружен не векторизатор (нет .transform).")
        return TextVectorizer(vec)
