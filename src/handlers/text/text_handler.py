from aiogram import Router, F
from aiogram.types import Message
from src.resources.messages import BotMessages

router = Router()


@router.message(F.text)
async def process_all_text(message: Message):
    """
    Обрабатывает все текстовые сообщения от пользователя.
    """
    await message.answer(
        BotMessages.all_text_msg
    )
