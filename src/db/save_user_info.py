import aiosqlite
from src.configs.config import ProjectConfig


async def save_info(user_id: int, username: str) -> None:
    async with aiosqlite.connect(ProjectConfig.DB_FILE_PATH) as conn:
        await conn.execute(
            """
            INSERT OR IGNORE INTO users (user_id, username)
            VALUES (?, ?)
            """,
            (user_id, username)
        )
        await conn.commit()
