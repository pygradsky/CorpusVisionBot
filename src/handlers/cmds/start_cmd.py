from aiogram import Router
from aiogram.types import Message
from aiogram.filters import CommandStart

from src.resources.messages import BotMessages
from src.db.save_user_info import save_info

router = Router()


@router.message(CommandStart())
async def process_start_cmd(message: Message):
    user_id = message.from_user.id
    username = message.from_user.username
    await save_info(user_id, username)

    await message.answer(
        BotMessages.start_cmd_msg
    )
