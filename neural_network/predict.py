import asyncio
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from src.configs.config import ProjectConfig


def _build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def _load_model():
    """
    Загружает модель из NN_FILE_PATH и возвращает (model, classes, transform).
    Вызывается один раз — результат кешируется в _MODEL_CACHE.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ProjectConfig.NN_FILE_PATH, map_location=device)
    classes = ckpt["classes"]

    model = _build_model(num_classes=len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    img_h, img_w = ckpt.get("img_size", ProjectConfig.IMG_SIZE)
    tf = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return model, classes, tf, device


_MODEL_CACHE: dict = {}


def _get_model():
    if not _MODEL_CACHE:
        model, classes, tf, device = _load_model()
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["classes"] = classes
        _MODEL_CACHE["tf"] = tf
        _MODEL_CACHE["device"] = device
    return (
        _MODEL_CACHE["model"],
        _MODEL_CACHE["classes"],
        _MODEL_CACHE["tf"],
        _MODEL_CACHE["device"],
    )


def _predict_sync(image_path: str) -> tuple[str, float]:
    """Синхронное предсказание. Вызывается из async-обёртки."""
    model, classes, tf, device = _get_model()

    img = Image.open(image_path).convert("RGB")
    tensor = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    pred_idx = probs.argmax().item()
    class_name = classes[pred_idx]
    confidence = probs[pred_idx].item()  # float 0.0 – 1.0

    return class_name, confidence


async def predict_image(image_path: str) -> tuple[str, float]:
    """
    Асинхронная обёртка для вызова из aiogram-хендлера.

    Возвращает:
        class_name  — str,   например "building_01"
        confidence  — float, например 0.97  (умножь на 100 чтобы получить %)

    Пример:
        class_name, confidence = await predict_image(file_path)
        confidence_pct = int(confidence * 100)   # → 97
    """
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _predict_sync, image_path)
    return result
