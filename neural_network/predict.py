import os
import asyncio
import threading
import logging
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.configs.config import ProjectConfig

_model = None
_model_lock = threading.Lock()


def _get_model():
    """
    Загружает модель один раз (thread-safe)
    """
    global _model
    with _model_lock:
        if _model is None:
            logging.info("Loading neural network model...")
            _model = load_model(ProjectConfig.NN_FILE_PATH)
            logging.info("Model loaded successfully")
        return _model


def _predict_sync(image_path: str) -> tuple:
    """
    СИНХРОННОЕ предсказание (вызывается в executor)
    """
    model = _get_model()

    img = image.load_img(
        image_path,
        target_size=ProjectConfig.IMG_SIZE
    )
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]

    class_index = int(np.argmax(preds))
    confidence = int(preds[class_index] * 100)

    class_names = sorted(
        os.listdir(ProjectConfig.DATASET_FOLDER_PATH)
    )
    class_name = class_names[class_index]

    return class_name, confidence


async def predict_image(image_path: str) -> tuple:
    """
    Асинхронное предсказание для Telegram-бота
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        _predict_sync,
        image_path
    )
