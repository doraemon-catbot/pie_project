"""파이류 7클래스 분류기"""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

MODEL_PATH = Path(__file__).parent.parent / "models/classifier/pie_classifier.pth"
CLASS_NAMES_PATH = Path(__file__).parent.parent / "models/classifier/class_names.txt"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PieClassifier:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.device = get_device()

        # 클래스명 로드
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.num_classes = len(self.class_names)

        # 모델 로드
        print(f"분류기 로딩 ({self.num_classes}클래스)...")
        self.model = models.mobilenet_v3_small(weights=None)
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features, self.num_classes
        )
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print("분류기 로딩 완료!")

        # 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.inference_mode()
    def classify(self, image: Image.Image) -> dict:
        """단일 이미지 분류, {"brand": str, "confidence": float} 반환"""
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        outputs = self.model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

        max_prob, max_idx = probs.max(0)
        confidence = float(max_prob)

        if confidence >= self.threshold:
            brand = self.class_names[int(max_idx)]
        else:
            brand = "미분류"

        return {"brand": brand, "confidence": confidence}

    @torch.inference_mode()
    def classify_batch(self, images: list[Image.Image]) -> list[dict]:
        """여러 이미지 분류"""
        if not images:
            return []

        tensors = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensors.append(self.transform(img))

        batch = torch.stack(tensors).to(self.device)
        outputs = self.model(batch)
        probs = torch.softmax(outputs, dim=1)

        results = []
        for prob in probs:
            max_prob, max_idx = prob.max(0)
            confidence = float(max_prob)

            if confidence >= self.threshold:
                brand = self.class_names[int(max_idx)]
            else:
                brand = "미분류"

            results.append({"brand": brand, "confidence": confidence})

        return results
