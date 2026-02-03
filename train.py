"""파이류 7클래스 분류기 학습 스크립트"""

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# 설정
DATA_DIR = Path("data/train")
MODEL_SAVE_PATH = Path("models/classifier/pie_classifier.pth")
CLASS_NAMES_PATH = Path("models/classifier/class_names.txt")
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 데이터 변환
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 데이터셋 로드
print("데이터 로딩 중...")
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)

print(f"클래스 수: {num_classes}")
print(f"클래스: {class_names}")
print(f"전체 이미지: {len(full_dataset)}장")

# Train/Val 분할
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Val 데이터셋은 augmentation 없이
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"학습: {train_size}장, 검증: {val_size}장")

# 모델 생성 (MobileNetV3 Small)
print("모델 생성 중...")
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
model = model.to(device)

# 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 학습
print(f"\n학습 시작 (Epochs: {EPOCHS})")
print("-" * 50)

best_acc = 0.0
for epoch in range(EPOCHS):
    # Train
    model.train()
    train_loss = 0.0
    train_correct = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()

    train_loss = train_loss / train_size
    train_acc = train_correct / train_size

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()

    val_loss = val_loss / val_size
    val_acc = val_correct / val_size

    scheduler.step()

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # Best 모델 저장
    if val_acc > best_acc:
        best_acc = val_acc
        MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

# 클래스명 저장
with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print("-" * 50)
print(f"학습 완료! Best Val Accuracy: {best_acc:.4f}")
print(f"모델 저장: {MODEL_SAVE_PATH}")
print(f"클래스명 저장: {CLASS_NAMES_PATH}")
