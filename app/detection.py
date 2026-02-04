"""DETR 기반 매대 상품 검출기"""

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

MODEL_ID = "isalia99/detr-resnet-50-sku110k"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ShelfDetector:
    def __init__(self, threshold: float = 0.5, max_det: int = 400):
        self.threshold = threshold
        self.max_det = max_det
        self.device = get_device()

        print(f"검출기 로딩: {MODEL_ID} ({self.device})...")
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForObjectDetection.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()
        print("검출기 로딩 완료!")

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> list[dict]:
        """이미지에서 상품 검출, [{"bbox": [x1,y1,x2,y2], "score": float}, ...] 반환"""
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[h, w]], device=self.device),
            threshold=self.threshold,
        )[0]

        boxes = results["boxes"].detach().cpu()
        scores = results["scores"].detach().cpu()
        labels = results["labels"].detach().cpu()

        # SKU Item (label == 1) 만 사용
        keep = labels == 1
        boxes, scores = boxes[keep], scores[keep]

        # 상위 max_det만 유지
        if len(scores) > self.max_det:
            idx = torch.argsort(scores, descending=True)[:self.max_det]
            boxes, scores = boxes[idx], scores[idx]

        # 좌표 클리핑
        boxes[:, 0::2].clamp_(0, w - 1)
        boxes[:, 1::2].clamp_(0, h - 1)

        detections = []
        for box, score in zip(boxes.tolist(), scores.tolist()):
            detections.append({
                "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                "score": float(score)
            })

        # 중첩 박스 제거 (박스 안의 박스 제거)
        detections = self._remove_nested_boxes(detections)

        return detections

    def _remove_nested_boxes(self, detections: list[dict], containment_threshold: float = 0.7) -> list[dict]:
        """
        한 박스가 다른 박스 안에 포함되어 있으면 제거.
        containment_threshold: 작은 박스의 면적 중 큰 박스와 겹치는 비율이 이 값 이상이면 제거
        """
        if len(detections) <= 1:
            return detections

        keep = [True] * len(detections)

        for i, det_i in enumerate(detections):
            if not keep[i]:
                continue
            box_i = det_i["bbox"]
            area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])

            for j, det_j in enumerate(detections):
                if i == j or not keep[j]:
                    continue
                box_j = det_j["bbox"]
                area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])

                # 겹치는 영역 계산
                x1 = max(box_i[0], box_j[0])
                y1 = max(box_i[1], box_j[1])
                x2 = min(box_i[2], box_j[2])
                y2 = min(box_i[3], box_j[3])

                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)

                    # 작은 박스가 큰 박스 안에 포함되어 있는지 확인
                    if area_i < area_j:
                        containment_ratio = intersection / area_i
                        if containment_ratio >= containment_threshold:
                            keep[i] = False
                            break
                    else:
                        containment_ratio = intersection / area_j
                        if containment_ratio >= containment_threshold:
                            keep[j] = False

        return [det for det, k in zip(detections, keep) if k]

    def crop_detections(self, image: Image.Image, detections: list[dict]) -> list[Image.Image]:
        """검출 영역을 crop하여 반환"""
        crops = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)
        return crops
