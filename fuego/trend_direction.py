"""Describe centroid changes in article coverage, not changes in real events."""

import math


def _unit_mean(rows):
    if not rows:
        return None
    if any(len(row) != 384 or any(not math.isfinite(x) for x in row) for row in rows):
        raise ValueError("Expected finite 384-dimensional vectors.")
    mean = [math.fsum(row[i] for row in rows)/len(rows) for i in range(384)]
    norm = math.hypot(*mean)
    return [x/norm for x in mean] if norm > 1e-12 else None


def direction(current, previous, topics):
    """Topics are name/vector pairs. Return up to three positive shift matches."""
    now, before = _unit_mean(current), _unit_mean(previous)
    if now is None or before is None:
        return {"description": "Insufficient temporal data.", "related_topics": []}
    cosine = max(-1.0, min(1.0, math.fsum(a*b for a, b in zip(now, before))))
    shift = [a-b for a, b in zip(now, before)]
    norm = math.hypot(*shift)
    if norm < 1e-8:
        return {"description": "No measurable centroid shift in these windows.", "related_topics": [],
                "centroid_cosine_distance": 0.0}
    hits = []
    for name, vector in topics:
        unit = _unit_mean([vector])
        if unit is None:
            continue
        score = max(-1.0, min(1.0, math.fsum(a*b for a, b in zip(shift, unit))/norm))
        if score > 0:
            hits.append({"name": name, "similarity": score})
    hits.sort(key=lambda item: item["similarity"], reverse=True)
    return {"description": "Article coverage shifted between the two publication windows.",
            "related_topics": hits[:3], "centroid_cosine_distance": 1-cosine}
