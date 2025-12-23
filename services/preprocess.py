import re


class TextPreprocessor:
    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^а-яёa-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
