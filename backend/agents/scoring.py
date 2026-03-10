from typing import List, Optional


TIER_1_KEYWORDS = [
    "nerf", "neural radiance", "gaussian splat", "3d gaussian",
    "novel view synthesis", "4d represent", "world model", "spatial foundation",
]

TIER_2_KEYWORDS = [
    "3d reconstruction", "scene represent", "embodied", "physics simul",
    "digital twin", "point cloud", "mesh", "volumetric", "radiance field",
    "view synthesis", "3d generat", "robot learning", "spatial ai",
]

TIER_3_KEYWORDS = [
    "3d", "depth estimation", "slam", "odometry", "pose estimation",
    "scene understanding", "object detection 3d", "lidar", "voxel",
]

NEGATIVE_KEYWORDS = [
    "medical imaging", "clinical", "pathology", "natural language",
    "text generation", "speech recognition", "drug discovery",
]


def score_researcher(
    publications: List[str],
    research_areas: List[str],
    abstract_texts: List[str],
) -> int:
    """Score a researcher 0-100 based on relevance to world models / 3D research.

    Args:
        publications: List of paper titles.
        research_areas: List of research area / theme strings.
        abstract_texts: List of paper abstracts (may contain empty strings).

    Returns:
        Integer score between 0 and 100.
    """
    # Combine all text for matching
    all_texts: List[str] = []
    for t in publications:
        all_texts.append(t.lower())
    for t in research_areas:
        all_texts.append(t.lower())
    for t in abstract_texts:
        if t:
            all_texts.append(t.lower())

    combined = " ".join(all_texts)

    raw_score = 0

    # Tier 1 — weight 3
    for kw in TIER_1_KEYWORDS:
        hits = combined.count(kw)
        raw_score += hits * 3

    # Tier 2 — weight 2
    for kw in TIER_2_KEYWORDS:
        hits = combined.count(kw)
        raw_score += hits * 2

    # Tier 3 — weight 1
    for kw in TIER_3_KEYWORDS:
        hits = combined.count(kw)
        raw_score += hits * 1

    # Negative — weight -2
    for kw in NEGATIVE_KEYWORDS:
        hits = combined.count(kw)
        raw_score += hits * (-2)

    # Normalize to 0-100.  Empirically, a very relevant researcher might
    # accumulate ~60-80 raw points across 5-10 papers+abstracts, so we
    # use 60 as our "100 %" reference and cap at 100.
    if raw_score <= 0:
        return 0

    normalized = int((raw_score / 60) * 100)
    return min(normalized, 100)
