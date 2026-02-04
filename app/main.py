"""Gradio 기반 파이류 점유율 분석 대시보드"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from app.detection import ShelfDetector
from app.classification import PieClassifier
from app.analysis import calculate_share, extract_brand

# 전역 모델
detector = None
classifier = None

# 브랜드별 색상
BRAND_COLORS = {
    "초코파이": "#8B4513",
    "참붕어빵": "#FF6347",
    "신카스타드": "#FFD700",
    "마켓오리얼브라우니": "#4B0082",
    "케익오뜨": "#FF69B4",
    "후레쉬베리": "#DC143C",
    "쉘위": "#00CED1",
    "오뜨": "#FF69B4",
    "미분류": "#808080",
}


def load_models():
    global detector, classifier
    if detector is None:
        detector = ShelfDetector()
    if classifier is None:
        classifier = PieClassifier()


def analyze_image(image_input):
    if image_input is None:
        return None, None, "이미지를 업로드해주세요."

    load_models()

    if isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    else:
        image = Image.open(image_input)

    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.thumbnail((1024, 1024))

    # 검출
    detections = detector.detect(image)
    if not detections:
        return np.array(image), None, "검출된 상품이 없습니다."

    # 분류
    crops = detector.crop_detections(image, detections)
    classifications = classifier.classify_batch(crops)

    # 검출 결과에 분류 결과 병합
    for det, cls in zip(detections, classifications):
        det["flavor"] = cls["flavor"]
        det["brand"] = extract_brand(cls["flavor"])
        det["confidence"] = cls["confidence"]

    # 점유율 계산
    share_result = calculate_share(detections)

    # 이미지에 bbox 그리기
    annotated_image = draw_boxes(image, detections)

    # 차트 생성
    chart = create_chart(share_result)

    # 결과 텍스트
    result_text = format_result(share_result)

    return np.array(annotated_image), chart, result_text


def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    draw = ImageDraw.Draw(image)

    # 한글 지원 폰트 로드 (Windows)
    try:
        font = ImageFont.truetype("malgun.ttf", 14)
    except:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        brand = det.get("brand", "미분류")
        flavor = det.get("flavor", "미분류")
        confidence = det.get("confidence", 0)
        color = BRAND_COLORS.get(brand, "#808080")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 브랜드명만 표시
        bbox = draw.textbbox((x1, y1 - 18), brand, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
        draw.text((x1, y1 - 18), brand, fill="white", font=font)

    return image


def create_chart(share_result: dict):
    brand_shares = share_result.get("brand_shares", {})

    if not brand_shares:
        return None

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(8, 6))

    labels = list(brand_shares.keys())
    sizes = list(brand_shares.values())
    colors = [BRAND_COLORS.get(label, "#808080") for label in labels]

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title(f"파이류 브랜드별 점유율 (총 {share_result['pie_count']}개)")

    plt.tight_layout()
    return fig


def format_result(share_result: dict) -> str:
    lines = [
        f"## 분석 결과",
        f"",
        f"**전체 검출**: {share_result['total_count']}개",
        f"**파이류**: {share_result['pie_count']}개 ({share_result['pie_share']}%)",
        f"",
        f"### 브랜드별 현황 (Facing / 면적)",
    ]

    for brand, count in sorted(share_result["brand_counts"].items(), key=lambda x: -x[1]):
        facing_share = share_result["brand_shares"].get(brand, 0)
        area_share = share_result.get("brand_areas", {}).get(brand, 0)
        lines.append(f"- **{brand}**: {count}개 (Facing {facing_share}% / 면적 {area_share}%)")

    # 맛별 상세
    if share_result.get("flavor_counts"):
        lines.append("")
        lines.append("### 맛별 상세")
        for flavor, count in sorted(share_result["flavor_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"- {flavor}: {count}개")

    return "\n".join(lines)


with gr.Blocks(title="오리온 파이류 점유율 분석") as demo:
    gr.Markdown("# 오리온 파이류 매대 점유율 분석")
    gr.Markdown("매대 사진을 업로드하면 파이류 브랜드별 점유율을 분석합니다.")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="매대 사진 업로드", type="filepath")
            analyze_btn = gr.Button("분석 시작", variant="primary")
            result_text = gr.Markdown(value="이미지를 업로드해주세요.")

        with gr.Column(scale=1):
            annotated_output = gr.Image(label="검출 결과")
            chart_output = gr.Plot(label="점유율 차트")

    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input],
        outputs=[annotated_output, chart_output, result_text]
    )

if __name__ == "__main__":
    demo.launch()
