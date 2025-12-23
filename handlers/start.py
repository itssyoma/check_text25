from maxapi.types import MessageCreated, Command


def register_start_handler(dp):
    @dp.message_created(Command("start"))
    async def start_handler(event: MessageCreated):
        await event.message.answer(
            "Привет! Пришли текст — я выполню предварительный анализ на токсичность/негативную риторику."
        )
