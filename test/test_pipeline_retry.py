"""Check the frozen retry policy, progress, and durable loss reports."""

from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from fuego.ai import AIConfig, AIError, NemotronClient
from fuego.pipeline import MockAI, Pipeline, PipelineConfig
from test.test_pipeline import TinyEmbedder, record


class PipelineRetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = io.StringIO()
        self.redirect = redirect_stderr(self.log)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def pipeline(self, client=None):
        return Pipeline(PipelineConfig(self.root, mock_ai=client is None), client=client, embedder=TinyEmbedder())

    def test_exactly_five_transport_calls_with_no_nested_retries(self):
        client = NemotronClient(AIConfig(api_key="test-key", timeout=120, max_retries=4))
        client._opener = Mock()
        client._opener.open.side_effect = TimeoutError()
        pipeline = self.pipeline(client)
        pipeline.enqueue([record()])
        with patch('fuego.ai.time.sleep') as sleep:
            report = pipeline.run_pending(refresh=False)
        self.assertEqual(client.config.timeout, 5)
        self.assertEqual(client.config.max_retries, 0)
        self.assertEqual(client._opener.open.call_count, 5)
        self.assertTrue(all(call.kwargs['timeout'] == 5 for call in client._opener.open.call_args_list))
        sleep.assert_not_called()
        self.assertEqual(report['summary']['loss_rate'], 1.0)
        self.assertEqual(report['failed_articles'][0]['attempts'], 5)
        self.assertEqual(report['failed_articles'][0]['error_stage'], 'article_ai')
        self.assertIn('attempt=5/5', self.log.getvalue())
        self.assertIn('LOST (input retained)', self.log.getvalue())
        self.assertNotIn('test-key', self.log.getvalue())

    def test_pipeline_overrides_environment_even_if_values_are_invalid(self):
        with patch.dict(os.environ, {'NVIDIA_API_KEY': 'test-key', 'NVIDIA_TIMEOUT': 'bad', 'NVIDIA_MAX_RETRIES': 'bad'}):
            pipeline = Pipeline(PipelineConfig(self.root), embedder=TinyEmbedder())
            client = pipeline._ai()
        self.assertEqual((client.config.timeout, client.config.max_retries), (5, 0))
        with patch.dict(os.environ, {'NVIDIA_API_KEY': 'test-key', 'NVIDIA_TIMEOUT': '17', 'NVIDIA_MAX_RETRIES': '2'}):
            standalone = AIConfig.from_env()
        self.assertEqual((standalone.timeout, standalone.max_retries), (17, 2))

    def test_fifth_attempt_can_succeed(self):
        pipeline = self.pipeline()
        real = pipeline.client.complete
        count = 0
        def reply(messages):
            nonlocal count
            count += 1
            if count < 5:
                raise AIError('offline')
            return real(messages)
        pipeline.client.complete = reply
        pipeline.enqueue([record()])
        report = pipeline.run_pending(refresh=False)
        self.assertEqual(count, 5)
        self.assertEqual(report['summary']['done'], 1)
        self.assertEqual(report['summary']['loss_rate'], 0)

    def test_loss_rate_excludes_filter_and_persists_both_reports(self):
        pipeline = self.pipeline()
        real = pipeline.client.complete
        def reply(messages):
            if json.loads(messages[-1]['content'])['metadata']['Record_ID'] == 'bad':
                raise AIError('offline')
            return real(messages)
        pipeline.client.complete = reply
        pipeline.enqueue([record('good'), record('bad'), dict(record('long'), Article_Text='x'*10001)])
        report = pipeline.run_pending(refresh=False)
        self.assertEqual(report['summary']['processed'], 2)
        self.assertEqual(report['summary']['filtered'], 1)
        self.assertEqual(report['summary']['loss_rate'], 0.5)
        self.assertEqual(json.loads(Path(report['report_path']).read_text()), report)
        self.assertEqual(json.loads((self.root/'reports/latest-work.json').read_text()), report)
        with sqlite3.connect(pipeline.store.db_path) as db:
            self.assertIsNotNone(db.execute("SELECT payload FROM jobs WHERE id='bad'").fetchone()[0])
        no_work = pipeline.run_pending(refresh=False)
        self.assertEqual(no_work['run_summary']['total'], 0)
        self.assertEqual(no_work['summary']['loss_rate'], 0.5)

    def test_refresh_failure_still_writes_report(self):
        pipeline = self.pipeline()
        pipeline.enqueue([record()])
        with patch.object(pipeline, 'refresh', side_effect=RuntimeError('index unavailable')):
            report = pipeline.run_pending()
        self.assertEqual(report['summary']['done'], 1)
        self.assertEqual(report['summary']['loss_rate'], 0)
        self.assertEqual(report['refresh_status'], 'failed')
        self.assertEqual(report['refresh_error'], 'RuntimeError')
        self.assertTrue(Path(report['report_path']).exists())

    def test_interrupt_writes_report_with_pending_not_lost(self):
        pipeline = self.pipeline()
        pipeline.client.complete = Mock(side_effect=KeyboardInterrupt())
        pipeline.enqueue([record()])
        with self.assertRaises(KeyboardInterrupt):
            pipeline.run_pending()
        report = json.loads((self.root/'reports/latest-work.json').read_text())
        self.assertTrue(report['interrupted'])
        self.assertEqual(report['summary']['pending'], 1)
        self.assertEqual(report['summary']['failed'], 0)

    def test_synthesis_uses_same_five_attempt_budget(self):
        client = NemotronClient(AIConfig(api_key='test-key', timeout=120, max_retries=4))
        client._opener = Mock()
        client._opener.open.side_effect = TimeoutError()
        pipeline = self.pipeline(client)
        article = {'summary': 'A bus trial.', 'title': 'Trial', 'source': 'Sample', 'published_at': 0}
        with patch('fuego.ai.time.sleep') as sleep:
            _, state = pipeline._synthesize(['a'], {'a': article}, 'Transit')
        self.assertEqual(state, 'failed')
        self.assertEqual(client._opener.open.call_count, 5)
        sleep.assert_not_called()

    def test_heartbeat_during_slow_embedding(self):
        pipeline = self.pipeline()
        embed = pipeline.embedder.embed
        def slow(text):
            threading.Event().wait(0.05)
            return embed(text)
        pipeline.embedder.embed = slow
        pipeline.enqueue([record()])
        with patch('fuego.pipeline.HEARTBEAT_SECONDS', 0.01):
            pipeline.run_pending(refresh=False)
        self.assertIn('RUNNING elapsed=', self.log.getvalue())
        self.assertIn('stage=embedding', self.log.getvalue())
        self.assertIn('SUMMARY run:', self.log.getvalue())

    def test_attempt_budget_is_frozen_and_old_pending_budget_handled(self):
        for count in (1, 3, 6):
            with self.assertRaises(ValueError):
                PipelineConfig(self.root, attempts=count)
        pipeline = self.pipeline()
        pipeline.enqueue([record()])
        with sqlite3.connect(pipeline.store.db_path) as db:
            db.execute('UPDATE jobs SET attempts=5')
        pipeline.client.complete = Mock()
        report = pipeline.run_pending(refresh=False)
        pipeline.client.complete.assert_not_called()
        self.assertEqual(report['failed_articles'][0]['error'], 'AttemptBudgetExhausted')

    def test_synthesis_loss_is_separate_from_article_loss(self):
        pipeline = self.pipeline()
        pipeline.enqueue([record()])
        with patch('fuego.pipeline.synthesize_topic', side_effect=AIError('offline')) as synthesize:
            report = pipeline.run_pending()
        self.assertEqual(report['summary']['loss_rate'], 0)
        failed = report['synthesis']['failed_buckets']
        self.assertTrue(failed)
        self.assertEqual(synthesize.call_count, 5*len(failed))
        self.assertEqual(report['synthesis']['ai_attempts'], synthesize.call_count)

    def test_old_job_schema_and_remaining_attempts_resume(self):
        path = self.root/'db/articles.sqlite3'
        path.parent.mkdir(parents=True)
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, payload TEXT, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, error TEXT, metadata TEXT NOT NULL)")
        pipeline = self.pipeline()
        pipeline.enqueue([record()])
        with sqlite3.connect(path) as db:
            db.execute('UPDATE jobs SET attempts=3')
        pipeline.client.complete = Mock(side_effect=AIError('offline'))
        report = pipeline.run_pending(refresh=False)
        self.assertEqual(pipeline.client.complete.call_count, 2)
        self.assertEqual(report['failed_articles'][0]['attempts'], 5)
        self.assertEqual(report['failed_articles'][0]['error_stage'], 'article_ai')

    def test_first_nonempty_batch_after_empty_refresh_builds_clusters(self):
        pipeline = self.pipeline()
        pipeline.refresh()
        pipeline.enqueue([record()])
        report = pipeline.run_pending()
        self.assertEqual(report['refresh_status'], 'done')
        self.assertEqual(len(pipeline.feed('cluster')['articles']), 1)

    def test_progress_is_flushed(self):
        pipeline = self.pipeline()
        with patch('builtins.print') as output:
            pipeline._info('working')
        self.assertTrue(output.call_args.kwargs['flush'])
