import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.configs.config import ProjectConfig, WAYS


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms():
    img_h, img_w = ProjectConfig.IMG_SIZE

    train_tf = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_tf, val_tf


def build_dataloaders(val_split: float = 0.2, num_workers: int = 4):
    """
    Ожидаемая структура датасета (DATASET_FOLDER_PATH):
        dataset/
            train/
                building_01/  img1.jpg ...
                building_02/  ...
                other/        ...
            val/              ← опционально, если нет — auto-split
                building_01/  ...
                ...

    Папки классов должны совпадать с ключами WAYS.
    """
    train_tf, val_tf = get_transforms()
    dataset_path = ProjectConfig.DATASET_FOLDER_PATH

    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")

    if os.path.isdir(val_dir):
        print("Найдена отдельная папка val/ — используем её.")
        train_ds = ImageFolder(train_dir, transform=train_tf)
        val_ds = ImageFolder(val_dir, transform=val_tf)
    else:
        print(f"Папки val/ нет — делаем auto-split ({int(val_split * 100)}% val).")
        root = train_dir if os.path.isdir(train_dir) else dataset_path
        full_ds = ImageFolder(root, transform=train_tf)
        val_size = int(len(full_ds) * val_split)
        train_size = len(full_ds) - val_size
        train_ds, val_ds = random_split(
            full_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        # Применяем val-transform к val-подмножеству
        val_ds.dataset = ImageFolder(root, transform=val_tf)

    classes = (train_ds.classes
               if hasattr(train_ds, "classes")
               else train_ds.dataset.classes)

    missing = set(WAYS.keys()) - set(classes)
    if missing:
        print(f"⚠  В датасете нет папок для классов: {missing}")

    train_loader = DataLoader(
        train_ds, batch_size=ProjectConfig.BATCH_SIZE,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=ProjectConfig.BATCH_SIZE,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    print(f"Классов: {len(classes)} → {classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")
    return train_loader, val_loader, classes


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """ResNet18 с заменённой головой под наши классы."""
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


class EarlyStopping:
    def __init__(self, patience: int = 7, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss: float) -> bool:
        if self.patience == 0:
            return False
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def final_evaluate(model, loader, device, classes):
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        preds = model(images.to(device)).argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

    print("\n── Classification Report ──")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(max(8, len(classes)), max(6, len(classes) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(ProjectConfig.NN_FOLDER_PATH, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix → {cm_path}")


def save_training_curves(history: dict):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(ep, history["train_loss"], label="Train")
    axes[0].plot(ep, history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(ep, history["train_acc"], label="Train")
    axes[1].plot(ep, history["val_acc"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(ProjectConfig.NN_FOLDER_PATH, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Кривые обучения → {path}")


def train():
    set_seed(42)

    # Создаём все нужные папки
    for folder in ProjectConfig.REQUIRED_FOLDERS:
        os.makedirs(folder, exist_ok=True)
    os.makedirs(ProjectConfig.NN_FOLDER_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"Датасет    : {ProjectConfig.DATASET_FOLDER_PATH}")
    print(f"Сохранение : {ProjectConfig.NEW_NN_FILE_PATH}\n")

    # ── Данные ──
    train_loader, val_loader, classes = build_dataloaders()

    # ── Модель ──
    model = build_model(num_classes=len(classes), pretrained=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Модель: ResNet18 | Параметров: {n_params:,}\n")

    # ── Loss / Optimizer / Scheduler ──
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=ProjectConfig.EPOCHS, eta_min=1e-6
    )

    # ── TensorBoard ──
    tb_dir = os.path.join(ProjectConfig.NN_FOLDER_PATH, "runs")
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"TensorBoard: tensorboard --logdir {tb_dir}\n")

    early_stop = EarlyStopping(patience=7)
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ── Цикл ──
    print("── Начало обучения ──")
    for epoch in range(1, ProjectConfig.EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, None, device, train=False)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{ProjectConfig.EPOCHS} | "
            f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
            f"Acc {tr_acc * 100:.2f}%/{vl_acc * 100:.2f}% | "
            f"LR {lr_now:.2e} | {elapsed:.1f}s"
        )

        writer.add_scalars("Loss", {"train": tr_loss, "val": vl_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": tr_acc, "val": vl_acc}, epoch)
        writer.add_scalar("LR", lr_now, epoch)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        # Сохраняем лучшую модель в NEW_NN_FILE_PATH, не трогая боевую
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": vl_acc,
                    "classes": classes,
                    "img_size": ProjectConfig.IMG_SIZE,
                },
                ProjectConfig.NEW_NN_FILE_PATH,
            )
            print(f"  ✓ Лучшая модель сохранена (val_acc={vl_acc * 100:.2f}%)")

        if early_stop(vl_loss):
            print(f"\nEarly stopping на эпохе {epoch}.")
            break

    writer.close()

    # ── Артефакты ──
    save_training_curves(history)

    print("\n── Финальная оценка лучшей модели ──")
    ckpt = torch.load(ProjectConfig.NEW_NN_FILE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    final_evaluate(model, val_loader, device, classes)

    print(f"\n✓ Готово! Лучшая val accuracy: {best_val_acc * 100:.2f}%")
    print(f"✓ Новая модель: {ProjectConfig.NEW_NN_FILE_PATH}")
    print(f"\nЧтобы заменить боевую модель:")
    print(f"  cp {ProjectConfig.NEW_NN_FILE_PATH} {ProjectConfig.NN_FILE_PATH}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    train()
