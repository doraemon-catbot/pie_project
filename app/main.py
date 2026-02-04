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
    "초코파이": "#D2691E", "참붕어빵": "#FF6347", "신카스타드": "#FFD700",
    "마켓오리얼브라우니": "#8B008B", "마켓오다쿠아즈": "#9932CC",
    "오뜨": "#FF1493", "후레쉬베리": "#DC143C", "쉘위": "#20B2AA",
    "쌀카스테라": "#DEB887", "ZERO": "#2F4F4F", "롯데 카스타드": "#DAA520",
    "몽쉘": "#A0522D", "찰떡파이": "#DB7093", "롱스": "#CD5C5C",
    "오예스": "#8B4513", "크림블": "#FFB6C1", "빅파이": "#FF4500",
    "미분류": "#808080",
}

# 회사별 색상 (오리온: 빨강, 롯데: 상아색, 해태크라운: 초록)
COMPANY_COLORS = {
    "오리온": "#DC2626",
    "롯데": "#D4A574",
    "해태크라운": "#16A34A",
    "기타": "#6B7280"
}


def load_models():
    global detector, classifier
    if detector is None:
        detector = ShelfDetector()
    if classifier is None:
        classifier = PieClassifier()


def analyze_image(image_input):
    if image_input is None:
        return None, None, "📷 이미지를 업로드해주세요."

    load_models()

    if isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input)
    else:
        image = Image.open(image_input)

    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.thumbnail((1280, 1280))

    # 검출
    detections = detector.detect(image)
    if not detections:
        return np.array(image), None, "❌ 검출된 상품이 없습니다."

    # 분류
    crops = detector.crop_detections(image, detections)
    classifications = classifier.classify_batch(crops)

    for det, cls in zip(detections, classifications):
        det["flavor"] = cls["flavor"]
        det["brand"] = extract_brand(cls["flavor"])
        det["confidence"] = cls["confidence"]

    # 쉘위 박스 확장 (왼쪽이 잘리는 문제 보정)
    w, h = image.size
    for det in detections:
        if det["brand"] == "쉘위":
            x1, y1, x2, y2 = det["bbox"]
            box_w = x2 - x1
            det["bbox"] = [
                int(max(0, x1 - box_w * 0.20)),  # 왼쪽 20% 확장
                y1,                               # 위 그대로
                x2,                               # 오른쪽 그대로
                y2                                # 아래 그대로
            ]

    # 점유율 계산
    share_result = calculate_share(detections)

    # 이미지에 bbox 그리기
    annotated_image = draw_boxes(image.copy(), detections)

    # 차트 생성
    chart = create_chart(share_result)

    # 결과 텍스트
    result_text = format_result(share_result)

    return np.array(annotated_image), chart, result_text


def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("malgun.ttf", 12)
    except:
        font = ImageFont.load_default()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        brand = det.get("brand", "미분류")
        color = BRAND_COLORS.get(brand, "#808080")

        # 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 라벨 배경
        label = brand
        bbox = draw.textbbox((x1, y1 - 16), label, font=font)
        padding = 2
        draw.rectangle(
            [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding],
            fill=color
        )
        draw.text((x1, y1 - 16), label, fill="white", font=font)

    return image


def create_chart(share_result: dict):
    company_shares = share_result.get("company_shares", {})
    company_areas = share_result.get("company_areas", {})
    if not company_shares:
        return None

    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), facecolor='#FAFAFA')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#FAFAFA')

    # Facing 점유율
    labels1 = list(company_shares.keys())
    sizes1 = list(company_shares.values())
    colors1 = [COMPANY_COLORS.get(label, "#6B7280") for label in labels1]

    wedges1, texts1, autotexts1 = ax1.pie(
        sizes1, labels=labels1, colors=colors1,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2),
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax1.set_title("Facing 점유율", fontsize=14, fontweight='bold', pad=15)

    # 면적 점유율
    labels2 = list(company_areas.keys())
    sizes2 = list(company_areas.values())
    colors2 = [COMPANY_COLORS.get(label, "#6B7280") for label in labels2]

    wedges2, texts2, autotexts2 = ax2.pie(
        sizes2, labels=labels2, colors=colors2,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2),
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax2.set_title("면적 점유율", fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout(pad=2)
    return fig


def format_result(share_result: dict) -> str:
    total = share_result['total_count']
    pie_count = share_result['pie_count']

    # HTML 기반 시각적 결과
    html = f"""
<div style="font-family: 'Malgun Gothic', sans-serif;">

<div style="display: flex; gap: 20px; margin-bottom: 20px;">
    <div style="background: #F3F4F6; padding: 15px 25px; border-radius: 10px; text-align: center;">
        <div style="font-size: 28px; font-weight: bold; color: #1F2937;">{total}</div>
        <div style="font-size: 12px; color: #6B7280;">전체 검출</div>
    </div>
    <div style="background: #FEF3C7; padding: 15px 25px; border-radius: 10px; text-align: center;">
        <div style="font-size: 28px; font-weight: bold; color: #D97706;">{pie_count}</div>
        <div style="font-size: 12px; color: #92400E;">파이류</div>
    </div>
</div>
"""

    # 회사별 점유율 (컬러 바)
    if share_result.get("company_counts"):
        html += '<div style="margin-bottom: 20px;"><strong>회사별 점유율</strong></div>'

        # Facing 점유율
        html += '<div style="margin-bottom: 15px;"><div style="font-size: 12px; color: #6B7280; margin-bottom: 5px;">Facing</div>'
        html += '<div style="display: flex; height: 35px; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">'

        for company in ["오리온", "롯데", "해태크라운", "기타"]:
            share = share_result["company_shares"].get(company, 0)
            if share > 0:
                color = COMPANY_COLORS.get(company, "#6B7280")
                text_color = "#FFF" if company != "롯데" else "#1F2937"
                html += f'<div style="width: {share}%; background: {color}; display: flex; align-items: center; justify-content: center; color: {text_color}; font-size: 11px; font-weight: bold;">{company} {share}%</div>'

        html += '</div></div>'

        # 면적 점유율
        html += '<div style="margin-bottom: 20px;"><div style="font-size: 12px; color: #6B7280; margin-bottom: 5px;">면적</div>'
        html += '<div style="display: flex; height: 35px; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">'

        for company in ["오리온", "롯데", "해태크라운", "기타"]:
            share = share_result.get("company_areas", {}).get(company, 0)
            if share > 0:
                color = COMPANY_COLORS.get(company, "#6B7280")
                text_color = "#FFF" if company != "롯데" else "#1F2937"
                html += f'<div style="width: {share}%; background: {color}; display: flex; align-items: center; justify-content: center; color: {text_color}; font-size: 11px; font-weight: bold;">{company} {share}%</div>'

        html += '</div></div>'

    # 브랜드별 (가로 테이블)
    if share_result.get("brand_counts"):
        brands = sorted(share_result["brand_counts"].items(), key=lambda x: -x[1])[:8]

        html += '<div style="margin-top: 15px;"><strong>브랜드별 상세</strong></div>'
        html += '<table style="width: 100%; margin-top: 10px; border-collapse: collapse; font-size: 12px;">'

        # 헤더 (브랜드명)
        html += '<tr style="background: #F9FAFB;">'
        html += '<td style="padding: 8px; border: 1px solid #E5E7EB; font-weight: bold;"></td>'
        for brand, _ in brands:
            html += f'<td style="padding: 8px; border: 1px solid #E5E7EB; text-align: center; font-weight: bold;">{brand}</td>'
        html += '</tr>'

        # 수량
        html += '<tr>'
        html += '<td style="padding: 8px; border: 1px solid #E5E7EB; background: #F9FAFB;">수량</td>'
        for brand, count in brands:
            html += f'<td style="padding: 8px; border: 1px solid #E5E7EB; text-align: center;">{count}</td>'
        html += '</tr>'

        # Facing %
        html += '<tr>'
        html += '<td style="padding: 8px; border: 1px solid #E5E7EB; background: #F9FAFB;">Facing</td>'
        for brand, _ in brands:
            facing = share_result["brand_shares"].get(brand, 0)
            html += f'<td style="padding: 8px; border: 1px solid #E5E7EB; text-align: center;">{facing}%</td>'
        html += '</tr>'

        # 면적 %
        html += '<tr>'
        html += '<td style="padding: 8px; border: 1px solid #E5E7EB; background: #F9FAFB;">면적</td>'
        for brand, _ in brands:
            area = share_result.get("brand_areas", {}).get(brand, 0)
            html += f'<td style="padding: 8px; border: 1px solid #E5E7EB; text-align: center;">{area}%</td>'
        html += '</tr>'

        html += '</table>'

    html += '</div>'
    return html


# Gradio UI
with gr.Blocks(
    title="파이류 점유율 분석",
    theme=gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="gray",
        neutral_hue="gray",
    )
) as demo:

    # 헤더
    gr.Markdown("# 파이류 매대 점유율 분석")

    with gr.Row():
        # 왼쪽: 입력
        with gr.Column(scale=1, min_width=300):
            image_input = gr.Image(
                label="매대 사진",
                type="filepath",
                height=250
            )
            analyze_btn = gr.Button(
                "🔍 분석하기",
                variant="primary",
                size="lg"
            )

        # 오른쪽: 결과 (HTML)
        with gr.Column(scale=1, min_width=400):
            result_text = gr.HTML(
                value='<div style="padding: 40px; text-align: center; color: #9CA3AF;">이미지를 업로드하고 분석하기 버튼을 클릭하세요</div>'
            )

    # 검출 결과 이미지
    annotated_output = gr.Image(
        label="검출 결과",
        show_label=True
    )

    # 차트
    chart_output = gr.Plot(label="회사별 점유율", visible=False)

    # 이벤트 연결
    analyze_btn.click(
        fn=analyze_image,
        inputs=[image_input],
        outputs=[annotated_output, chart_output, result_text]
    )


if __name__ == "__main__":
    demo.launch()
