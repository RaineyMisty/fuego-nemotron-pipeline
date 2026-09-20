"""Check a known change between unit centroids."""
from fuego.trend_direction import direction


def main():
    x, y = [1.0]+[0.0]*383, [0.0, 1.0]+[0.0]*382
    result = direction([y], [x], [("New topic", y), ("Old topic", x)])
    if result["centroid_cosine_distance"] != 1 or result["related_topics"][0]["name"] != "New topic":
        raise ValueError("Wrong direction calculation.")
    print("PASS direction: orthogonal centroids and correct shift topic")


if __name__ == "__main__":
    main()
