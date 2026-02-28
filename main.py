import aiosqlite
import os
import asyncio
import logging

from aiogram import Bot, Dispatcher
from dotenv import load_dotenv

from src.utils.setup_project import create_environment
from src.resources.errors import BotErrors

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")


async def main():
    try:
        await create_environment()
    except (OSError, aiosqlite.Error) as e:
        logging.error(f"{BotErrors.ENVIRONMENT_ERROR}: {e}")
        return

    bot = Bot(token=BOT_TOKEN)
    dp = Dispatcher()

    await dp.start_polling(bot)


if __name__ == '__main__':
    try:
        logging.basicConfig(level=logging.INFO)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Досрочный выход")
