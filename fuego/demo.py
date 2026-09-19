"""Use simple offline rules to test the data flow. No model is called."""

import re


class WorkflowDemoClient:
    mode = "demo"
    model = "offline-rules-no-model"

    def complete(self, system, payload):
        task = payload["task"]
        if task == "extract_scheme":
            article = payload["article"]
            note = article["text"][:500]
            return {"themes": ["demo"], "people": [], "organizations": [], "locations": [],
                    "tone": "unknown", "summary": note,
                    "evidence": [{"article_id": article["id"], "quote": note}]}
        if task == "query_buckets":
            return {"bucket_ids": self.match(payload["query"], payload["buckets"])}
        if task == "route_schemes":
            return {"assignments": [{"article_id": a["id"], "bucket_ids": self.match(
                (a["scheme"]["summary"] or "") + " " + " ".join(a["scheme"]["themes"]), payload["buckets"])}
                                    for a in payload["articles"]]}
        if task == "digest":
            return {"bullets": [{"text": a["text"][:500], "evidence": [
                {"article_id": a["id"], "quote": a["text"][:500]}]} for a in payload["articles"][:5]]}
        raise ValueError("Unknown demo task.")

    @staticmethod
    def match(content, buckets):
        aliases = {"education": ["school", "university", "campus", "education", "学校", "大学"],
                   "ai": ["ai", "artificial intelligence", "machine learning", "人工智能"],
                   "robotics": ["robot", "robotics", "机器人"],
                   "economy": ["economy", "jobs", "business"],
                   "transport": ["bus", "transit", "transport"],
                   "health": ["health", "medical"], "community": ["community", "festival"]}
        content = content.casefold()
        def found(word):
            return bool(re.search(r"(?<!\w)" + re.escape(word.casefold()) + r"(?!\w)", content))
        return [b["id"] for b in buckets if any(found(word) for word in
                [b["id"], b["name"], *aliases.get(b["id"], [])])]
