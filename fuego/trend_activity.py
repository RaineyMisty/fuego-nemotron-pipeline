"""Compare article counts in two equal publication-time windows."""

DAY_MS = 86400000


def activity(timestamps, *, as_of, window_ms=DAY_MS, coverage_start=None):
    if type(as_of) is not int or type(window_ms) is not int or window_ms <= 0:
        raise ValueError("Use an integer time and a positive integer window.")
    if any(type(t) is not int for t in timestamps):
        raise ValueError("Publication times must be integer milliseconds.")
    current = sum(as_of-window_ms < t <= as_of for t in timestamps)
    prior = sum(as_of-2*window_ms < t <= as_of-window_ms for t in timestamps)
    if coverage_start is None or coverage_start > as_of-2*window_ms:
        return {"current_count": current, "previous_count": None, "change_ratio": None, "status": "insufficient_data"}
    if prior == 0:
        return {"current_count": current, "previous_count": 0, "change_ratio": None,
                "status": "new" if current else "stable"}
    return {"current_count": current, "previous_count": prior, "change_ratio": (current-prior)/prior,
            "status": "increasing" if current > prior else "decreasing" if current < prior else "stable"}
