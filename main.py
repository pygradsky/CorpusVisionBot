import aiosqlite
import os
import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from dotenv import load_dotenv

from src.utils.setup_environment import create_environment
from src.resources.errors import BotErrors
from src.handlers import __all_routers__

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")


async def main():
    try:
        await create_environment()
    except (OSError, aiosqlite.Error) as e:
        logging.error(f"{BotErrors.ENVIRONMENT_ERROR}: {e}")
        return

    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()

    for router in __all_routers__:
        dp.include_router(router)

    await dp.start_polling(bot)


if __name__ == '__main__':
    try:
        logging.basicConfig(level=logging.INFO)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Досрочный выход")
