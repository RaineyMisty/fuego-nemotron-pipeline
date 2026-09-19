"""Check article input and model output."""

from datetime import datetime, timezone
import math
from urllib.parse import urlsplit

from .ingest import InputError, check
from .pipeline import keys, text, items, require, validate_summary


def fields(value, required, optional=()):
    check(isinstance(value, dict) and set(required) <= set(value)
          and not (set(value) - set(required) - set(optional)),
          "Invalid fields. Required: " + ", ".join(required) + ". Optional: " + ", ".join(optional) + ".")


def string(value, name, limit):
    check(isinstance(value, str) and bool(value.strip()) and len(value) <= limit,
          f"{name} must have 1-{limit} text characters.")
    return value


def timestamp(value, name):
    if value is None:
        return None
    string(value, name, 40)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        check(parsed.tzinfo is not None, name + " needs a timezone.")
        return parsed.astimezone(timezone.utc).isoformat()
    except (ValueError, OverflowError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError(name + " must be an ISO timestamp.") from None


def normalize_articles(value):
    fields(value, ("articles",))
    rows = value["articles"]
    check(isinstance(rows, list) and 1 <= len(rows) <= 100, "Send 1-100 articles.")
    articles, ids = [], set()
    for row in rows:
        fields(row, ("id", "url"), ("title", "source", "text", "published_at", "observed_at", "gdelt"))
        aid = string(row["id"], "id", 100)
        check(aid not in ids, "Article IDs must be unique in a batch.")
        ids.add(aid)
        url = string(row["url"], "url", 2048)
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in ("http", "https") and parsed.hostname and not parsed.username
            _ = parsed.port
        except ValueError:
            valid = False
        check(valid and not any(c.isspace() for c in url), "Use an HTTP(S) URL without credentials or whitespace.")
        published = timestamp(row.get("published_at"), "published_at")
        observed = timestamp(row.get("observed_at"), "observed_at")
        check(published or observed, "Provide published_at or observed_at.")
        content = row.get("text")
        if content is not None:
            string(content, "text", 12000)
        title, source = row.get("title"), row.get("source")
        if title is not None:
            string(title, "title", 500)
        if source is not None:
            string(source, "source", 200)
        gdelt = row.get("gdelt", {})
        fields(gdelt, (), ("theme_codes", "tone"))
        codes = gdelt.get("theme_codes", [])
        check(isinstance(codes, list) and len(codes) <= 100, "Use at most 100 theme codes.")
        for code in codes:
            string(code, "theme code", 200)
        tone = gdelt.get("tone")
        check(tone is None or (type(tone) in (int, float) and math.isfinite(tone) and -100 <= tone <= 100),
              "GDELT tone must be null or a finite number from -100 to 100.")
        check(content is not None or "gdelt" in row, "Supply article text or GDELT metadata.")
        articles.append({"id": aid, "url": url, "title": title, "source": source, "text": content,
                         "published_at": published, "observed_at": observed,
                         "date": (published or observed)[:10],
                         "time_basis": "published_at" if published else "observed_at",
                         "gdelt": {"theme_codes": sorted(set(codes)), "tone": tone}})
    return articles


def validate_scheme(value, article):
    keys(value, ("themes", "people", "organizations", "locations", "tone", "summary", "evidence"), "scheme")
    for name in ("themes", "people", "organizations", "locations"):
        entries = items(value[name], name, 20)
        for entry in entries:
            text(entry, name, 150)
        require(len(entries) == len(set(entries)), name + " must not contain duplicates.")
    require(value["tone"] in ("positive", "negative", "neutral", "mixed", "unknown"), "Invalid tone.")
    validate_summary({"bullets": [{"text": value["summary"], "evidence": value["evidence"]}]}, [article])


def validate_selection(value, buckets):
    keys(value, ("bucket_ids",), "query result")
    selected = items(value["bucket_ids"], "bucket_ids", len(buckets))
    allowed = {b["id"] for b in buckets}
    for bid in selected:
        require(isinstance(bid, str) and bid in allowed, "Unknown bucket ID.")
    require(len(selected) == len(set(selected)), "Duplicate bucket ID.")


def validate_digest(value, articles):
    keys(value, ("bullets",), "digest")
    if value["bullets"] != []:
        validate_summary(value, articles)
