import os
import logging
import aiosqlite

from aiogram import Router, F
from aiogram.types import Message

from neural_network.predict import predict_image
from src.configs.config import WAYS
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

        replied_msg = await message.reply(BotMessages.downloaded_photo_msg)

        class_name, confidence = await predict_image(file_path)
        confidence = int(confidence* 100)
        corpus_number = class_name.split("_")[-1]

        if class_name == "other":
            msg = (
                "✅ Готово! Вот что удалось определить:\n\n"
                f"• Изображение не является корпусом\n"
                f"• Уверенность: {confidence}%"
            )
        else:
            address = WAYS[class_name]
            msg = (
                "✅ Готово! Вот что удалось определить:\n\n"
                f"• Корпус №{corpus_number}\n"
                f"• Уверенность: {confidence}%\n"
                f"• Адрес: (скопируйте текст ниже)\n"
                f"» <code>{address}</code>"
            )
        await replied_msg.edit_text(msg)

    except OSError as e:
        msg = BotErrors.FILE_PROCESSING_ERROR
        logging.error(f"{msg}: {e}")
        await message.answer(
            f"{msg}\n"
            f"{BotMessages.help_msg}"
        )
    except aiosqlite.Error as e:
        msg = BotErrors.AIOSQLITE_ERROR
        logging.error(f"{msg}: {e}")
        await message.answer(
            f"{msg}\n"
            f"{BotMessages.help_msg}"
        )
