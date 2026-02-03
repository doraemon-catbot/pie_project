# pie_project

오리온 파이류 매대 점유율 분석 도구

## 개요

매대 사진에서 오리온 파이류 제품을 검출하고, 브랜드별 점유율을 대시보드로 시각화합니다.

## 대상 브랜드 (8개)

1. 초코파이
2. 참붕어빵
3. 카스타드
4. 쌀 카스테라
5. 마켓오 리얼브라우니
6. 케익오뜨
7. 후레쉬베리
8. 쉘위

## 주요 기능

- 매대 사진 업로드 → 상품 자동 검출 (DETR)
- 파이류 8개 브랜드 분류 (MobileNetV3)
- 오리온 파이 전체 점유율 계산
- 브랜드별 점유율 차트 시각화
- Gradio 기반 대시보드

## 프로젝트 구조

```
pie_project/
├── README.md
├── requirements.txt
├── app/
│   ├── main.py           # Gradio 대시보드
│   ├── detection.py      # DETR 검출기
│   ├── classification.py # 파이 분류기
│   └── analysis.py       # 점유율 계산
├── models/
│   └── classifier/       # 학습된 분류 모델
├── data/
│   ├── train/            # 학습 데이터 (브랜드별 폴더)
│   └── val/              # 검증 데이터
└── docs/
    └── PRD.md            # 제품 요구사항 문서
```

## 설치

```bash
pip install -r requirements.txt
```

## 실행

```bash
python app/main.py
```

## 기술 스택

- 검출: DETR ResNet-50 (SKU110K)
- 분류: MobileNetV3 Small
- UI: Gradio
- 프레임워크: PyTorch, Transformers

## 관련 프로젝트

- `pingu`: 57개 브랜드 점유율 분석 (보류)
- `facing-analyzer`: 초코파이 facing 카운터
- `crop_Project`: 학습 데이터 crop 도구
