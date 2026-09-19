"""Validate long-format GDELT rows and merge them by Record_ID."""

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import math
import re
from urllib.parse import urlsplit

COLUMNS = ("Record_ID", "Article_Link", "Theme", "Tone")
MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 100_000
MAX_RECORDS = 1_000
TONE_FIELDS = ("tone", "positive_score", "negative_score", "polarity",
               "activity_reference_density", "self_group_reference_density", "word_count")


class InputError(ValueError):
    def __init__(self, message, code="invalid_input", status=422):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class Record:
    id: str
    url: str
    observed_at: str
    themes: tuple[tuple[str, int], ...]
    tone: tuple[float, ...]

    @property
    def theme_codes(self):
        return sorted({code for code, _ in self.themes})


@dataclass(frozen=True)
class Batch:
    records: tuple[Record, ...]
    input_rows: int
    duplicate_theme_rows: int
    blank_theme_rows: int


def check(condition, message):
    if not condition:
        raise InputError(message)


def decode_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            check(key not in result, "Duplicate JSON object key.")
            result[key] = value
        return result

    def invalid_constant(value):
        raise InputError("JSON numbers must be finite.")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (ValueError, RecursionError) as exc:
        if isinstance(exc, InputError):
            raise
        raise InputError("Invalid JSON.") from None


def parse_input(raw: bytes, media_type: str) -> Batch:
    if len(raw) > MAX_BYTES:
        raise InputError("Payload exceeds 8 MiB.", "payload_too_large", 413)
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InputError("Input must use UTF-8.") from None
    if media_type == "text/csv":
        reader = csv.DictReader(io.StringIO(content, newline=""), strict=True)
        try:
            check(reader.fieldnames is not None and len(reader.fieldnames) == len(COLUMNS)
                  and set(reader.fieldnames) == set(COLUMNS),
                  "CSV header must contain exactly Record_ID, Article_Link, Theme, Tone.")
            return normalize_rows(reader)
        except csv.Error:
            raise InputError("Malformed CSV. Quote Theme and Tone fields containing commas.") from None
    if media_type == "application/json":
        value = decode_json(content)
        check(isinstance(value, dict) and set(value) == {"records"},
              'JSON body must be {"records": [{"Record_ID": ..., "Article_Link": ..., "Theme": ..., "Tone": ...}]}.')
        check(isinstance(value["records"], list), "records must be an array.")
        return normalize_rows(value["records"])
    raise InputError("Use text/csv or application/json.", "unsupported_media_type", 415)


def normalize_rows(rows) -> Batch:
    groups = {}
    count = duplicates = blanks = 0
    for count, row in enumerate(rows, 1):
        check(count <= MAX_ROWS, f"At most {MAX_ROWS} rows are allowed.")
        prefix = f"Row {count}: "
        check(isinstance(row, dict) and set(row) == set(COLUMNS), prefix + "invalid columns.")
        check(all(isinstance(row[k], str) for k in COLUMNS), prefix + "all fields must be strings.")
        rid, url, theme, tone_raw = (row[k].strip() for k in COLUMNS)
        check(bool(re.fullmatch(r"[0-9]{14}-[0-9]+", rid)) and len(rid) <= 100,
              prefix + "Record_ID must use YYYYMMDDhhmmss-number.")
        try:
            observed = datetime.strptime(rid[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            raise InputError(prefix + "Record_ID has an invalid timestamp.") from None
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in ("http", "https") and parsed.hostname and not parsed.username
            _ = parsed.port
        except ValueError:
            valid = False
        check(valid and len(url) <= 2048 and not any(c.isspace() for c in url),
              prefix + "Article_Link must be an HTTP(S) URL without credentials or whitespace.")
        check(len(tone_raw) <= 512, prefix + "Tone is too long.")
        try:
            tone = tuple(float(v) for v in tone_raw.split(","))
        except ValueError:
            raise InputError(prefix + "Tone must contain seven comma-separated numbers.") from None
        check(len(tone) == 7 and all(math.isfinite(v) for v in tone),
              prefix + "Tone must contain seven finite numbers.")
        check(-100 <= tone[0] <= 100 and all(0 <= v <= 100 for v in tone[1:6])
              and tone[6] >= 0 and tone[6].is_integer(), prefix + "Tone values are outside the allowed ranges.")
        entry = None
        if theme:
            check(len(theme) <= 300, prefix + "Theme is too long.")
            parts = theme.rsplit(",", 1)
            check(len(parts) == 2 and bool(re.fullmatch(r"[A-Za-z0-9_]+", parts[0]))
                  and bool(re.fullmatch(r"[0-9]{1,12}", parts[1])),
                  prefix + "Theme must be CODE,offset or an empty string.")
            entry = (parts[0], int(parts[1]))
        else:
            blanks += 1
        if rid not in groups:
            check(len(groups) < MAX_RECORDS, f"At most {MAX_RECORDS} unique records are allowed.")
            groups[rid] = {"url": url, "tone": tone, "observed": observed, "themes": set()}
        group = groups[rid]
        check(group["url"] == url and group["tone"] == tone,
              prefix + "the same Record_ID has conflicting Article_Link or Tone values.")
        if entry is not None:
            if entry in group["themes"]:
                duplicates += 1
            group["themes"].add(entry)
            check(len(group["themes"]) <= 4096, prefix + "too many theme occurrences for one record.")
    check(count > 0, "Send at least one data row.")
    records = tuple(Record(rid, g["url"], g["observed"], tuple(sorted(g["themes"])), g["tone"])
                    for rid, g in sorted(groups.items()))
    return Batch(records, count, duplicates, blanks)
