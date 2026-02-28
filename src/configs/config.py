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


WAYS = {
    "building_01": "127550 г. Москва, Лиственничная аллея, д.4А",
    "building_02": "127550 г. Москва, Лиственничная аллея, д. 4",
    "building_03": "127550 г. Москва, Лиственничная аллея, д.3",
    "building_04": "127550 г. Москва, ул. Пасечная, д. 2",
    "building_05": "127550 г. Москва, ул. Тимирязевская, д. 48",
    "building_07": "неизвестен",
    "building_08": "127550 г. Москва, Тимирязевская улица, д. 47",
    "building_09": "127550 г. Москва, ул. Тимирязевская, д. 52",
    "building_10": "127550 г. Москва, Тимирязевская улица, д. 49",
    "other": "неизвестен",
}
