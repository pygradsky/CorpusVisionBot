import os

PROJECT_FOLDER_PATH = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)


class ProjectConfig:
    # Данные
    DATA_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "data")
    DOWNLOADS_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "downloads")
    DATASET_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "dataset")

    # Базы Данных
    DB_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, "db")
    DB_FILE_PATH = os.path.join(DB_FOLDER_PATH, "users.db")

    # Нейросеть
    NN_FOLDER_PATH = os.path.join(PROJECT_FOLDER_PATH, "neural_network")
    NN_FILE_PATH = os.path.join(NN_FOLDER_PATH, "model.keras")
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20

    # Папки для создания
    REQUIRED_FOLDERS = [
        DATA_FOLDER_PATH,
        DOWNLOADS_FOLDER_PATH,
        DB_FOLDER_PATH,
        NN_FOLDER_PATH,
    ]
