import joblib


class TextClassifier:
    def __init__(self, model):
        self.model = model

    def predict(self, X) -> int:
        # возвращаем 0/1
        return int(self.model.predict(X)[0])

    def predict_proba(self, X) -> float:
        # если у модели нет predict_proba — вернём NaN-совместимое значение
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(X)[0][1])
        return float("nan")

    @staticmethod
    def load(path: str) -> "TextClassifier":
        m = joblib.load(path)
        if not hasattr(m, "predict"):
            raise TypeError(f"Из {path} загружена не модель (нет .predict).")
        return TextClassifier(m)
