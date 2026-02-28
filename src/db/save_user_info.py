import aiosqlite
from src.configs.config import ProjectConfig


async def save_info(user_id: int, username: str) -> None:
    """
    Сохраняет данные пользователя в БД
    """
    async with aiosqlite.connect(ProjectConfig.DB_FILE_PATH) as conn:
        await conn.execute(
            """
            INSERT OR IGNORE INTO users (user_id, username, photos_count)
            VALUES (?, ?, ?)
            """,
            (user_id, username, 0)
        )
        await conn.commit()
