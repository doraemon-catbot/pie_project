"""점유율 계산 모듈"""

from collections import Counter

# 브랜드 → 회사 매핑
BRAND_TO_COMPANY = {
    # 오리온
    "초코파이": "오리온",
    "참붕어빵": "오리온",
    "신카스타드": "오리온",
    "마켓오리얼브라우니": "오리온",
    "마켓오다쿠아즈": "오리온",
    "오뜨": "오리온",
    "후레쉬베리": "오리온",
    "쉘위": "오리온",
    "쌀카스테라": "오리온",
    # 롯데
    "ZERO": "롯데",
    "롯데 카스타드": "롯데",
    "몽쉘": "롯데",
    "찰떡파이": "롯데",
    # 해태크라운
    "롱스": "해태크라운",
    "오예스": "해태크라운",
    "크림블": "해태크라운",
}


def extract_brand(flavor_name: str) -> str:
    """맛 이름에서 브랜드 추출 (언더바 또는 공백 앞부분)"""
    if "_" in flavor_name:
        return flavor_name.split("_")[0]
    # "ZERO 초콜릿칩쿠키" 같은 공백 구분 이름 처리
    if " " in flavor_name:
        return flavor_name.split(" ")[0]
    return flavor_name


def get_company(brand: str) -> str:
    """브랜드에서 회사 추출"""
    return BRAND_TO_COMPANY.get(brand, "기타")


def calculate_share(detections: list[dict]) -> dict:
    """
    검출 결과에서 점유율 계산 (맛 → 브랜드 → 회사 집계)

    Args:
        detections: [{"bbox": [...], "flavor": str, "confidence": float}, ...]

    Returns:
        {
            "total_count": int,
            "pie_count": int,
            "pie_share": float,
            "brand_counts": {"브랜드": count, ...},
            "brand_shares": {"브랜드": share%, ...},
            "company_counts": {"회사": count, ...},
            "company_shares": {"회사": share%, ...},
            "flavor_counts": {"맛": count, ...}
        }
    """
    total_count = len(detections)

    if total_count == 0:
        return {
            "total_count": 0,
            "pie_count": 0,
            "pie_share": 0.0,
            "brand_counts": {},
            "brand_shares": {},
            "company_counts": {},
            "company_shares": {},
            "company_areas": {},
            "flavor_counts": {}
        }

    # 전체 검출 사용 (미분류도 "기타" 회사로 포함)
    pie_detections = detections
    pie_count = len(pie_detections)

    # 전체 중 파이 비율
    pie_share = (pie_count / total_count) * 100

    # 맛별 카운트
    flavors = [d["flavor"] for d in pie_detections]
    flavor_counts = dict(Counter(flavors))

    # 브랜드별 카운트 (맛에서 브랜드 추출하여 집계)
    brands = [extract_brand(d["flavor"]) for d in pie_detections]
    brand_counts = dict(Counter(brands))

    # 회사별 카운트
    companies = [get_company(extract_brand(d["flavor"])) for d in pie_detections]
    company_counts = dict(Counter(companies))

    # 브랜드별 점유율 (파이 중에서)
    brand_shares = {}
    if pie_count > 0:
        for brand, count in brand_counts.items():
            brand_shares[brand] = (count / pie_count) * 100

    # 회사별 점유율
    company_shares = {}
    if pie_count > 0:
        for company, count in company_counts.items():
            company_shares[company] = (count / pie_count) * 100

    # 면적 계산
    total_area = 0
    brand_areas = {}
    company_areas_raw = {}

    for d in detections:
        bbox = d.get("bbox", [0, 0, 0, 0])
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        total_area += area

    for d in pie_detections:
        bbox = d.get("bbox", [0, 0, 0, 0])
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        brand = extract_brand(d["flavor"])
        company = get_company(brand)
        brand_areas[brand] = brand_areas.get(brand, 0) + area
        company_areas_raw[company] = company_areas_raw.get(company, 0) + area

    # 브랜드별 면적 점유율
    pie_area = sum(brand_areas.values())
    brand_area_shares = {}
    if pie_area > 0:
        for brand, area in brand_areas.items():
            brand_area_shares[brand] = (area / pie_area) * 100

    # 회사별 면적 점유율
    company_area_shares = {}
    if pie_area > 0:
        for company, area in company_areas_raw.items():
            company_area_shares[company] = (area / pie_area) * 100

    return {
        "total_count": total_count,
        "pie_count": pie_count,
        "pie_share": round(pie_share, 1),
        "brand_counts": brand_counts,
        "brand_shares": {k: round(v, 1) for k, v in brand_shares.items()},
        "company_counts": company_counts,
        "company_shares": {k: round(v, 1) for k, v in company_shares.items()},
        "company_areas": {k: round(v, 1) for k, v in company_area_shares.items()},
        "flavor_counts": flavor_counts,
        "brand_areas": {k: round(v, 1) for k, v in brand_area_shares.items()},
        "total_area": total_area,
        "pie_area": pie_area
    }
