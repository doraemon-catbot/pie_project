"""점유율 계산 모듈"""

from collections import Counter


def extract_brand(flavor_name: str) -> str:
    """맛 이름에서 브랜드 추출 (언더바 앞부분)"""
    if "_" in flavor_name:
        return flavor_name.split("_")[0]
    return flavor_name


def calculate_share(detections: list[dict]) -> dict:
    """
    검출 결과에서 점유율 계산 (맛 → 브랜드 집계)

    Args:
        detections: [{"bbox": [...], "flavor": str, "confidence": float}, ...]

    Returns:
        {
            "total_count": int,
            "pie_count": int,
            "pie_share": float,
            "brand_counts": {"브랜드": count, ...},
            "brand_shares": {"브랜드": share%, ...},
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
            "flavor_counts": {}
        }

    # 파이류만 필터 (미분류 제외)
    pie_detections = [d for d in detections if d.get("flavor") != "미분류"]
    pie_count = len(pie_detections)

    # 전체 중 파이 비율
    pie_share = (pie_count / total_count) * 100

    # 맛별 카운트
    flavors = [d["flavor"] for d in pie_detections]
    flavor_counts = dict(Counter(flavors))

    # 브랜드별 카운트 (맛에서 브랜드 추출하여 집계)
    brands = [extract_brand(d["flavor"]) for d in pie_detections]
    brand_counts = dict(Counter(brands))

    # 브랜드별 점유율 (파이 중에서)
    brand_shares = {}
    if pie_count > 0:
        for brand, count in brand_counts.items():
            brand_shares[brand] = (count / pie_count) * 100

    # 면적 계산
    total_area = 0
    brand_areas = {}
    for d in detections:
        bbox = d.get("bbox", [0, 0, 0, 0])
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        total_area += area

    for d in pie_detections:
        bbox = d.get("bbox", [0, 0, 0, 0])
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        brand = extract_brand(d["flavor"])
        brand_areas[brand] = brand_areas.get(brand, 0) + area

    # 브랜드별 면적 점유율
    pie_area = sum(brand_areas.values())
    brand_area_shares = {}
    if pie_area > 0:
        for brand, area in brand_areas.items():
            brand_area_shares[brand] = (area / pie_area) * 100

    return {
        "total_count": total_count,
        "pie_count": pie_count,
        "pie_share": round(pie_share, 1),
        "brand_counts": brand_counts,
        "brand_shares": {k: round(v, 1) for k, v in brand_shares.items()},
        "flavor_counts": flavor_counts,
        "brand_areas": {k: round(v, 1) for k, v in brand_area_shares.items()},
        "total_area": total_area,
        "pie_area": pie_area
    }
