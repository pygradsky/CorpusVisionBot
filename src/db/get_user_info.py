import aiosqlite
from src.configs.config import ProjectConfig


async def get_info(user_id: int) -> list:
    """
    Получает все данные пользователя из БД
    """
    async with aiosqlite.connect(ProjectConfig.DB_FILE_PATH) as conn:
        cursor = await conn.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = await cursor.fetchall()
        return row
