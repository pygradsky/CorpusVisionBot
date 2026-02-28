import os
import aiosqlite
from src.configs.config import ProjectConfig


async def create_environment() -> None:
    """Подготавливает окружение проекта (папки + БД)."""
    await create_folders()
    await create_table()


async def create_folders() -> None:
    """Создаёт необходимые папки для корректной работы проекта."""
    for path in ProjectConfig.REQUIRED_FOLDERS:
        os.makedirs(path, exist_ok=True)


async def create_table() -> None:
    """Создаёт БД для корректной работы проекта."""
    async with aiosqlite.connect(ProjectConfig.DB_FILE_PATH) as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            user_name TEXT NOT NULL,
            join_date TEXT
            )
            """
        )
        await conn.commit()
