import logging

import aiosqlite

from aiogram import Router
from aiogram.types import Message
from aiogram.filters import CommandStart

from src.resources.messages import BotMessages
from src.resources.errors import BotErrors
from src.db.save_user_info import save_info
from src.utils.setup_user_env import create_user_env

router = Router()


@router.message(CommandStart())
async def process_start_cmd(message: Message) -> None:
    """
    Обрабатывает команду /start
    """
    user_id = message.from_user.id
    username = message.from_user.username

    try:
        await create_user_env(user_id)
        await save_info(user_id, username)
    except (OSError, aiosqlite.Error) as e:
        logging.error(
            f"{BotErrors.USER_ENVIRONMENT_ERROR}: {e}"
        )
        return

    await message.answer(
        BotMessages.start_cmd_msg
    )
