# Structure
fuego/
├── __init__.py
├── __main__.py
├── contracts.py
│
├── article_input.py
├── article_processing.py
├── embedding.py
├── map_store.py
├── named_buckets.py
├── query_buckets.py
├── topic_clusters.py
├── topic_synthesis.py
├── trend_activity.py
├── trend_direction.py
├── write_output.py
│
└── pipeline.py

tests/
├── unit/
│   ├── test_article_input.py
│   ├── test_article_processing.py
│   ├── test_embedding.py
│   ├── test_map_store.py
│   ├── test_named_buckets.py
│   ├── test_query_buckets.py
│   ├── test_topic_clusters.py
│   ├── test_topic_synthesis.py
│   ├── test_trend_activity.py
│   ├── test_trend_direction.py
│   └── test_write_output.py
│
└── integration/
    └── test_pipeline.py