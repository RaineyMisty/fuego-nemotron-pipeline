"""Check two known publication windows."""
from fuego.trend_activity import activity


def main():
    result = activity([5, 15, 16], as_of=20, window_ms=10, coverage_start=0)
    if result != {"current_count": 2, "previous_count": 1, "change_ratio": 1.0, "status": "increasing"}:
        raise ValueError("Wrong activity calculation.")
    print("PASS activity: 1 to 2 articles, change ratio 1.0")


if __name__ == "__main__":
    main()
