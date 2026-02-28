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
async def process_media(message: Message):
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

        await message.reply(
            BotMessages.downloaded_photo_msg
        )
    except OSError as e:
        logging.error(
            f"{BotErrors.OSE_ERROR}: {e}"
        )
        return
    except aiosqlite.Error as e:
        logging.error(
            f"{BotErrors.AIOSQLITE_ERROR}: {e}"
        )
        return
