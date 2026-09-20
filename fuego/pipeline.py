"""Coordinate durable offline jobs and independent news-feed requests."""

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
import threading
import time

from fuego.ai import NemotronClient, DEFAULT_MODEL
from fuego.article_input import ArticleInput, prepare_article
from fuego.article_processing import STOP_WORDS
from fuego.embedding import Embedder, MODEL_NAME
from fuego.map_store import MapStore
from fuego.named_buckets import NamedBuckets
from fuego.query_buckets import QueryBuckets
from fuego.topic_clusters import TopicClusters
from fuego.topic_synthesis import synthesize_topic
from fuego.trend_activity import activity, DAY_MS
from fuego.trend_direction import direction
from fuego.write_output import build_output


@dataclass(frozen=True)
class PipelineConfig:
    state_dir: Path
    mock_ai: bool = False
    mock_query: str | None = None
    model_cache: str | None = None
    local_files_only: bool = False
    min_score: float = 0.4
    cluster_count: int = 20
    attempts: int = 3
    window_ms: int = DAY_MS

    def __post_init__(self):
        if type(self.mock_ai) is not bool or type(self.local_files_only) is not bool:
            raise ValueError("Test and cache flags must be booleans.")
        if self.mock_query is not None and (not isinstance(self.mock_query, str) or not self.mock_query.strip()):
            raise ValueError("mock_query must be nonempty text.")
        if type(self.min_score) not in (int, float) or not math.isfinite(self.min_score) or not -1 <= self.min_score <= 1:
            raise ValueError("min_score must be in [-1, 1].")
        for value in (self.cluster_count, self.attempts, self.window_ms):
            if type(value) is not int or value <= 0:
                raise ValueError("Counts and window size must be positive integers.")


class MockAI:
    """Return deterministic test data. This is not a language model."""

    def complete(self, messages):
        data = json.loads(messages[-1]["content"])
        if "article" in data:
            title = data["metadata"].get("Title", "Sample article")
            words = re.findall(r"[A-Za-z][A-Za-z'-]*", title+" "+data["article"])
            terms, seen = [], set()
            for word in words:
                key = word.casefold()
                if key not in STOP_WORDS and key not in seen and len(word) > 2:
                    terms.append(word[:80]); seen.add(key)
                if len(terms) == 20:
                    break
            while len(terms) < 20:
                terms.append(f"sample term {len(terms)+1}")
            result = {"keywords": terms, "summary": "[MOCK AI] "+title[:300]+". "+data["article"][:550]}
        else:
            articles = data["articles"]
            result = {"title": "Sample topic synthesis",
                      "summary": "[MOCK AI] "+" ".join(a["summary"][:220] for a in articles)[:3500]}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}]}


class Pipeline:
    def __init__(self, config, *, client=None, embedder=None):
        self.config = config
        self.root = Path(config.state_dir)
        self.lock = threading.RLock()
        self.client = client if client is not None else (MockAI() if config.mock_ai else None)
        self.ai_model = getattr(getattr(client, "config", None), "model", os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL))
        self.embedder = embedder if embedder is not None else Embedder(
            cache_folder=config.model_cache, local_files_only=config.local_files_only)
        self.store = ArticleInput(self.root/"db"/"articles.sqlite3", client=self.client)
        self.map = MapStore(self.root/"map")
        self.named = NamedBuckets(self.map, embedder=self.embedder)
        self.queries = QueryBuckets(self.map, embedder=self.embedder)
        self.clusters = TopicClusters(self.map, k=config.cluster_count)
        with self._db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, payload TEXT, status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0, error TEXT, metadata TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS prepared (kind TEXT PRIMARY KEY, response TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS syntheses (key TEXT PRIMARY KEY, response TEXT NOT NULL);
            ''')
            signature = json.dumps({"embedding": MODEL_NAME, "backend": "fastembed-onnxruntime", "mock_ai": config.mock_ai, "ai_model": self.ai_model}, sort_keys=True)
            old = db.execute("SELECT value FROM settings WHERE key='signature'").fetchone()
            if old and old[0] != signature:
                raise ValueError("State mode or embedding model differs. Use another state directory.")
            db.execute("INSERT OR IGNORE INTO settings VALUES ('signature', ?)", (signature,))
            db.commit()

    def _db(self):
        return closing(sqlite3.connect(self.store.db_path))

    def _ai(self):
        if self.client is None:
            self.client = NemotronClient()
            self.store.client = self.client
        return self.client

    @staticmethod
    def _normalize(record):
        result = dict(record)
        date = result.get("Publication_Date")
        if isinstance(date, str) and date.isdigit():
            result["Publication_Date"] = int(date)
        for key in ("Source_Name", "Tone", "Title", "Article_Link", "People", "Organizations", "Themes"):
            if result.get(key) is None:
                result[key] = ""
        return result

    def enqueue(self, records):
        if not isinstance(records, list) or len(records) > 10000:
            raise ValueError("Input must be a list of at most 10000 articles.")
        normalized = []
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("Record_ID"), str) or not record["Record_ID"].strip():
                raise ValueError("Every queued article needs a Record_ID.")
            record = self._normalize(record)
            encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
            if len(encoded) > 2_000_000:
                raise ValueError("A record exceeds the queue input limit.")
            normalized.append((record["Record_ID"], encoded, json.dumps({k: v for k, v in record.items() if k != "Article_Text"})))
        with self._db() as db:
            inserted = 0
            for article_id, payload, metadata in normalized:
                cursor = db.execute("INSERT OR IGNORE INTO jobs(id,payload,status,metadata) VALUES (?,?,'pending',?)",
                                    (article_id, payload, metadata))
                inserted += cursor.rowcount
            db.commit()
        return {"queued": inserted, "duplicates": len(records)-inserted}

    def retry_failed(self):
        with self.lock, self._db() as db:
            count = db.execute("UPDATE jobs SET status='pending',attempts=0,error=NULL WHERE status='failed'").rowcount
            db.commit()
        return count

    def status(self):
        with self._db() as db:
            jobs = [{"id": row[0], "status": row[1], "attempts": row[2], "error": row[3]}
                    for row in db.execute("SELECT id,status,attempts,error FROM jobs ORDER BY id")]
            count = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        with self._db() as db:
            dirty = db.execute("SELECT value FROM settings WHERE key='dirty'").fetchone()
            ready = db.execute("SELECT COUNT(*) FROM prepared").fetchone()[0] == 2
        return {"articles": count, "mock_ai": self.config.mock_ai, "jobs": jobs,
                "feeds_stale": bool(dirty and dirty[0] == "1") or not ready}

    def run_pending(self, *, refresh=True):
        # The lock also protects the shared embedder and the JSON map writer.
        with self.lock:
            with self._db() as db:
                jobs = db.execute("SELECT id,payload,attempts FROM jobs WHERE status='pending' ORDER BY rowid").fetchall()
            for article_id, payload, attempts in jobs:
                for attempt in range(attempts+1, self.config.attempts+1):
                    try:
                        record = json.loads(payload)
                        prepared = prepare_article(record)
                        if prepared is not None and self.store.get(article_id) is None:
                            self.store.client = self._ai()
                        result = self.store.ingest(record)
                        if result["status"] != "filtered":
                            article = self.store.get(article_id)
                            if article_id not in self.map.get_map()["article_ids"]:
                                self.map.add(article_id, self.embedder.embed(article["semantic_text"]))
                        with self._db() as db:
                            db.execute("UPDATE jobs SET status=?,payload=NULL,attempts=?,error=NULL WHERE id=?",
                                       ("filtered" if result["status"] == "filtered" else "done", attempt, article_id))
                            db.execute("INSERT OR REPLACE INTO settings VALUES ('dirty', '1')")
                            db.commit()
                        break
                    except Exception as exc:
                        with self._db() as db:
                            db.execute("UPDATE jobs SET status=?,attempts=?,error=? WHERE id=?",
                                       ("failed" if attempt == self.config.attempts else "pending",
                                        attempt, type(exc).__name__, article_id))
                            db.commit()
                        if attempt < self.config.attempts:
                            time.sleep(0.1 if self.config.mock_ai else min(2**(attempt-1), 8))
            if refresh:
                self.refresh()
        return self.status()

    def _articles(self):
        snapshot = self.map.get_map()
        vectors = dict(zip(snapshot["article_ids"], snapshot["vectors"]))
        articles = {key: self.store.get(key) for key in vectors}
        if any(value is None for value in articles.values()):
            raise ValueError("Map contains an article missing from SQLite.")
        return articles, vectors

    def _synthesize(self, ids, articles, name):
        if not ids:
            return {"title": name[:120], "summary": "No matching articles."}, "empty"
        inputs = [{"summary": articles[key]["summary"], "metadata": {"title": articles[key]["title"],
                   "source": articles[key]["source"], "published_at": articles[key]["published_at"]}} for key in ids[:10]]
        key = hashlib.sha256(json.dumps([inputs, self.config.mock_ai, self.ai_model], sort_keys=True).encode()).hexdigest()
        with self._db() as db:
            cached = db.execute("SELECT response FROM syntheses WHERE key=?", (key,)).fetchone()
        if cached:
            return json.loads(cached[0]), "ready"
        for attempt in range(self.config.attempts):
            try:
                result = synthesize_topic(inputs, client=self._ai())
                with self._db() as db:
                    db.execute("INSERT OR REPLACE INTO syntheses VALUES (?, ?)", (key, json.dumps(result)))
                    db.commit()
                return result, "ready"
            except Exception:
                if attempt+1 < self.config.attempts:
                    time.sleep(0.1 if self.config.mock_ai else 1)
        return {"title": name[:120], "summary": "Summary unavailable. Source articles are listed below."}, "failed"

    def _package(self, kind, groups, articles, vectors, *, as_of=None):
        groups = [(key, name, description, sorted(hits, key=lambda h: (-h["score"], h["article_id"])))
                  for key, name, description, hits in groups]
        timestamp = int(time.time()*1000) if as_of is None else as_of
        if type(timestamp) is not int:
            raise ValueError("as_of must be integer milliseconds.")
        topic_vectors = self.named.load_vectors() if vectors else []
        topics = [(d["name"], v) for d, v in zip(self.named.definitions(), topic_vectors)]
        coverage = min((a["published_at"] for a in articles.values()), default=None)
        specs, statuses = [], {}
        for key, name, description, hits in groups:
            result, state = self._synthesize([h["article_id"] for h in hits[:10]], articles, name)
            specs.append({"id": key, "description": description, "synthesis": result})
            statuses[key] = state
        with tempfile.TemporaryDirectory(prefix="fuego-response-") as folder:
            snapshot_path = Path(folder)/"articles.sqlite3"
            with self._db() as source, closing(sqlite3.connect(snapshot_path)) as target:
                source.backup(target)
                target.execute("DELETE FROM article_buckets")
                for key, _, _, hits in groups:
                    target.executemany("INSERT INTO article_buckets VALUES (?, ?, ?)",
                                       [(h["article_id"], key, h["score"]) for h in hits])
                target.commit()
            with self._db() as db:
                metadata = {row[0]: json.loads(row[1]) for row in db.execute("SELECT id,metadata FROM jobs WHERE status='done'")}
            response = build_output(specs, db_path=snapshot_path, request_type=kind, embeddings=vectors,
                                    metadata=metadata, generated_at=datetime.fromtimestamp(timestamp/1000, timezone.utc))
        for bucket, (_, _, _, hits) in zip(response["buckets"], groups):
            ids = [h["article_id"] for h in hits]
            times = [articles[key]["published_at"] for key in ids]
            bucket["activity"] = activity(times, as_of=timestamp, window_ms=self.config.window_ms, coverage_start=coverage)
            current = [vectors[key] for key in ids if timestamp-self.config.window_ms < articles[key]["published_at"] <= timestamp]
            previous = [vectors[key] for key in ids if timestamp-2*self.config.window_ms < articles[key]["published_at"] <= timestamp-self.config.window_ms]
            bucket["direction"] = direction(current, previous, topics) if bucket["activity"]["status"] != "insufficient_data" else direction([], [], [])
            bucket["summary_status"] = statuses[bucket["id"]]
        response["parameters"].update(mock_ai=self.config.mock_ai, min_score=self.config.min_score,
                                      window_ms=self.config.window_ms, mock_query=self.config.mock_query is not None)
        response["warnings"] = ["AI replies are simulated. This output is not an AI quality result."] if self.config.mock_ai else []
        response["warnings"] += [f"Synthesis failed for {key}." for key, state in statuses.items() if state == "failed"]
        response["stats"]["indexed_articles"] = len(vectors)
        return response

    def refresh(self, *, force_clusters=False, as_of=None):
        with self.lock:
            articles, vectors = self._articles()
            if force_clusters:
                self.clusters.update()
            else:
                self.clusters.update_if_needed()
            definitions = self.named.definitions()
            selected = self.named.query(min_score=self.config.min_score)
            batches = self.map.search_many(self.named.load_vectors(), top_k=max(1, len(vectors))) if vectors else [[] for _ in definitions]
            groups = []
            links = {key: [] for key in articles}
            for definition, hits in zip(definitions, batches):
                hits = [h for h in hits if h["score"] >= self.config.min_score]
                # Keep the named module's selected IDs as a checked integration boundary.
                if set(selected[definition["id"]]) != {h["article_id"] for h in hits[:10]}:
                    raise ValueError("Named bucket selection differs from map scores.")
                groups.append((definition["id"], definition["name"], definition["prototype_text"], hits))
                for hit in hits:
                    links[hit["article_id"]].append({"bucket_id": definition["id"], "similarity": hit["score"]})
            for key, values in links.items():
                self.store.set_buckets(key, values)
            fixed = self._package("fixed", groups, articles, vectors, as_of=as_of)
            clustered = self.clusters.load()
            cluster_groups = []
            for index, center in enumerate(clustered["centers"]):
                norm = math.hypot(*center)
                ids = [key for key, label in clustered["assignments"].items() if label == index and key in vectors]
                hits = [{"article_id": key, "score": max(-1.0, min(1.0, math.fsum(a*b for a, b in zip(vectors[key], center))/norm)) if norm else 0.0} for key in ids]
                hits.sort(key=lambda hit: hit["score"], reverse=True)
                cluster_groups.append((f"cluster-{index}", f"Topic cluster {index+1}", "Articles grouped by K-means.", hits))
            clusters = self._package("cluster", cluster_groups, articles, vectors, as_of=as_of)
            clusters["stats"]["pending_cluster_articles"] = len(set(vectors)-set(clustered["assignments"]))
            with self._db() as db:
                for kind, response in (("fixed", fixed), ("cluster", clusters)):
                    db.execute("INSERT OR REPLACE INTO prepared VALUES (?, ?)", (kind, json.dumps(response, allow_nan=False)))
                db.execute("INSERT OR REPLACE INTO settings VALUES ('dirty', '0')")
                db.commit()
            return {"fixed": fixed, "cluster": clusters}

    def feed(self, kind):
        if kind not in ("fixed", "cluster"):
            raise ValueError("Unknown prepared feed.")
        with self._db() as db:
            row = db.execute("SELECT response FROM prepared WHERE kind=?", (kind,)).fetchone()
        if row is None:
            raise LookupError("Feed is not ready. Run offline processing first.")
        return json.loads(row[0])

    def query(self, topic=None, *, as_of=None):
        topic = self.config.mock_query if topic is None else topic
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("A nonempty topic is required.")
        with self.lock:
            articles, vectors = self._articles()
            ids = self.queries.query(topic, min_score=self.config.min_score)
            hits = self.map.search(self.embedder.embed(topic), top_k=10) if vectors else []
            hits = [h for h in hits if h["article_id"] in ids]
            key = "query-"+hashlib.sha256(topic.strip().encode()).hexdigest()[:16]
            return self._package("query", [(key, topic, topic, hits)], articles, vectors, as_of=as_of)
