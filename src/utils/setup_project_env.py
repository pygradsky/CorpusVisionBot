import os
import aiosqlite
from src.configs.config import ProjectConfig


async def create_project_env() -> None:
    """
    Подготавливает окружение проекта (папки + БД).
    """
    await _create_folders()
    await _create_table()


async def _create_folders() -> None:
    """
    Создаёт необходимые папки для корректной работы проекта.
    """
    for path in ProjectConfig.REQUIRED_FOLDERS:
        os.makedirs(path, exist_ok=True)


async def _create_table() -> None:
    """
    Создаёт БД для корректной работы проекта.
    """
    async with aiosqlite.connect(ProjectConfig.DB_FILE_PATH) as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            photos_count INTEGER,
            join_date TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await conn.commit()
