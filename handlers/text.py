from pathlib import Path

from maxapi import F
from maxapi.types import MessageCreated

from services.preprocess import TextPreprocessor
from services.vectorizer import TextVectorizer
from services.classifier import TextClassifier
from services.response import ResultFormatter


VECTORIZER_PATH = Path("models/vectorizer.pkl")
MODEL_PATH = Path("models/model.pkl")

# Грузим один раз при импорте (быстро и стабильно)
pre = TextPreprocessor()
fmt = ResultFormatter()

if not VECTORIZER_PATH.exists() or not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Не найдены файлы модели. Сначала запусти: python train_model.py\n"
        f"Ожидаю файлы:\n- {VECTORIZER_PATH}\n- {MODEL_PATH}"
    )

vec = TextVectorizer.load(str(VECTORIZER_PATH))
clf = TextClassifier.load(str(MODEL_PATH))


def register_text_handler(dp):
    @dp.message_created(F.message.body.text)
    async def text_handler(event: MessageCreated):
        text = (event.message.body.text or "").strip()

        if not text:
            await event.message.answer("Пришли непустой текст 🙂")
            return

        if len(text) > 5000:
            await event.message.answer("Текст слишком длинный. Сократи до 5000 символов.")
            return

        cleaned = pre.clean(text)
        X = vec.transform([cleaned])
        pred = clf.predict(X)
        proba = clf.predict_proba(X)

        await event.message.answer(fmt.format(pred=pred, proba=proba))
