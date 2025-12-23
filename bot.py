import asyncio
import os

from maxapi import Bot, Dispatcher
from handlers import register_handlers


async def main():
    token = os.getenv("MAX_BOT_TOKEN")
    if not token:
        raise RuntimeError("Установи переменную окружения MAX_BOT_TOKEN")

    bot = Bot(token=token)
    dp = Dispatcher()

    register_handlers(dp)

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
