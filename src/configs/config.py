import os

PROJECT_FOLDER_PATH = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)


class ProjectConfig:
    DATA_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "data")
    DOWNLOADS_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "downloads")

    DB_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "db")
    DB_FILE_PATH = os.path.join(DB_FOLDER_PATH, "users.db")

    NN_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "neural_network")
    NN_FILE_PATH = os.path.join(NN_FOLDER_PATH, "model.keras")

    REQUIRED_FOLDERS = [
        DATA_FOLDER_PATH,
        DOWNLOADS_FOLDER_PATH,
        DB_FOLDER_PATH,
        NN_FOLDER_PATH,
    ]
