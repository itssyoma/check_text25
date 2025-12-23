import math


class ResultFormatter:
    def format(self, pred: int, proba: float | None = None) -> str:
        if pred == 1:
            msg = "Результат: обнаружены признаки токсичной/негативной риторики."
        else:
            msg = "Результат: признаки токсичной/негативной риторики не выявлены."

        if proba is not None and not (isinstance(proba, float) and math.isnan(proba)):
            msg += f"\nУверенность модели: {proba:.2f}"

        msg += "\n\nВажно: результат носит предварительный характер и не является экспертным заключением."
        return msg
