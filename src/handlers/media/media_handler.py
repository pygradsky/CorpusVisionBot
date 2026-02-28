import asyncio
import os
import logging
import aiosqlite

from aiogram import Router, F
from aiogram.types import Message

from src.configs.config import ProjectConfig
from src.resources.errors import BotErrors
from src.resources.messages import BotMessages

router = Router()


@router.message(F.photo)
async def process_media(message: Message) -> None:
    """
    Обрабатывает изображение, полученное от пользователя
    """
    user_id = message.from_user.id
    user_photos_path = os.path.join(ProjectConfig.DOWNLOADS_FOLDER_PATH, str(user_id), "photos")
    try:
        largest_photo = message.photo[-1]
        file_name = "photo.jpg"
        file_path = os.path.join(user_photos_path, file_name)
        await message.bot.download(largest_photo, destination=file_path)

        replied_msg = await message.reply(
            BotMessages.downloaded_photo_msg
        )
        return
    except OSError as e:
        msg = BotErrors.OSE_ERROR
        logging.error(f"{msg}: {e}")
        await message.answer(msg)
    except aiosqlite.Error as e:
        msg = BotErrors.AIOSQLITE_ERROR
        logging.error(f"{msg}: {e}")
        await message.answer(msg)
