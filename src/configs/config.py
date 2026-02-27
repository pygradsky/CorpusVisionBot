import os
from dataclasses import dataclass

PROJECT_FOLDER_PATH = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)


@dataclass
class ProjectConfig:
    # Данные
    data_folder_path: str = os.path.join(PROJECT_FOLDER_PATH, "data")
    downloads_folder_path: str = os.path.join(data_folder_path, "downloads")

    # Базы данных
    db_folder_path: str = os.path.join(data_folder_path, "db")
    db_file_path: str = os.path.join(db_folder_path, "users")

    # Сохраненные модели
    checkpoints_folder_path: str = os.path.join(data_folder_path, "checkpoints_folder_path")

    # Папки создаваемые при отправке команды /start
    folders_to_setup = [
        data_folder_path,
        downloads_folder_path,
        db_folder_path,
        checkpoints_folder_path,
    ]
