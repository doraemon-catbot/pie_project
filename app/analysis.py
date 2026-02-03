"""점유율 계산 모듈"""

from collections import Counter


def calculate_share(detections: list[dict]) -> dict:
    """
    검출 결과에서 점유율 계산

    Args:
        detections: [{"bbox": [...], "brand": str, "confidence": float}, ...]

    Returns:
        {
            "total_count": int,
            "pie_count": int,
            "pie_share": float,
            "brand_counts": {"브랜드": count, ...},
            "brand_shares": {"브랜드": share%, ...}
        }
    """
    total_count = len(detections)

    if total_count == 0:
        return {
            "total_count": 0,
            "pie_count": 0,
            "pie_share": 0.0,
            "brand_counts": {},
            "brand_shares": {}
        }

    # 파이류만 필터 (미분류 제외)
    pie_detections = [d for d in detections if d.get("brand") != "미분류"]
    pie_count = len(pie_detections)

    # 전체 중 파이 비율
    pie_share = (pie_count / total_count) * 100

    # 브랜드별 카운트
    brands = [d["brand"] for d in pie_detections]
    brand_counts = dict(Counter(brands))

    # 브랜드별 점유율 (파이 중에서)
    brand_shares = {}
    if pie_count > 0:
        for brand, count in brand_counts.items():
            brand_shares[brand] = (count / pie_count) * 100

    return {
        "total_count": total_count,
        "pie_count": pie_count,
        "pie_share": round(pie_share, 1),
        "brand_counts": brand_counts,
        "brand_shares": {k: round(v, 1) for k, v in brand_shares.items()}
    }
