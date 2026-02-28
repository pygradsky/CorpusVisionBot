import os
from src.configs.config import ProjectConfig


async def create_user_env(user_id: int) -> None:
    """
    Подготавливает окружение пользователя (папки)
    """
    user_folder_path = os.path.join(ProjectConfig.DOWNLOADS_FOLDER_PATH, str(user_id))
    user_photos_path = os.path.join(user_folder_path, "photos")
    user_achievements_path = os.path.join(user_folder_path, "achievements")

    user_required_folders = [
        user_folder_path,
        user_photos_path,
        user_achievements_path,
    ]
    for path in user_required_folders:
        os.makedirs(path, exist_ok=True)
