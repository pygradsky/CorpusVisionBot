import os
from src.configs.config import ProjectConfig


async def create_folders() -> None:
    """Создаёт необходимые папки для работы проекта."""
    for path in ProjectConfig.required_folders:
        os.makedirs(path, exist_ok=True)
