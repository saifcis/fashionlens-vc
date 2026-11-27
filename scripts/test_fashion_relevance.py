"""Regression tests for fashion relevance. Run with the local CLIP cache and catalog."""

import io
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import Mock, patch
from uuid import UUID

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.embeddings.clip_provider import CLIPProvider
from app.search.engine import SearchEngine
from ui import discovery
from app.search.relevance import (
    FashionRelevance, RelevanceBlocked, RelevanceMode, configure_relevance_logging, logger,
)


NON_FASHION_SCORES = {
    "category_scores": {"clothing": 0.2, "logos": 0.3},
    "fashion_category": "clothing", "non_fashion_category": "logos",
    "margin": -0.1, "fashion_cosine": 0.2, "non_fashion_cosine": 0.3,
}


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.provider = Mock(spec=CLIPProvider)
        self.provider.MODEL_NAME = CLIPProvider.MODEL_NAME
        self.provider.encode_image.return_value = np.ones(512)
        self.relevance = FashionRelevance(self.provider, RelevanceMode.SOFT)
        self.relevance.classifier = Mock()

    def test_default_and_explicit_mode(self):
        with patch.dict(os.environ, {"FASHION_LENS_RELEVANCE_MODE": " enforce "}):
            self.assertEqual(RelevanceMode.configured(), RelevanceMode.ENFORCE)
        with patch.dict(os.environ, clear=True):
            self.assertEqual(RelevanceMode.configured(), RelevanceMode.SOFT)

    def test_frozen_prompt_parity(self):
        path = Path("experiments/clip_domain_prompts.json")
        if not path.is_file():
            self.skipTest("Archived evaluation prompts are excluded from Docker")
        frozen = json.loads(path.read_text())
        self.assertEqual(self.relevance.config["prompts"], frozen["prompts"])
        self.assertEqual(self.relevance.config["zero_shot_margin"], frozen["rules"]["zero_shot_margin"])

    def test_modes_and_override(self):
        self.relevance.classifier.score.return_value = NON_FASHION_SCORES
        for mode, decision in (
            (RelevanceMode.OFF, "continue"), (RelevanceMode.SHADOW, "continue"),
            (RelevanceMode.SOFT, "confirm"), (RelevanceMode.ENFORCE, "reject"),
        ):
            with self.subTest(mode=mode):
                self.relevance.mode = mode
                query = self.relevance.prepare("private-path.png")
                self.assertEqual(self.relevance.decision(query, "visual"), decision)
                if mode == RelevanceMode.OFF:
                    self.assertIsNone(query.relevance.scores)
                    self.relevance.classifier.score.assert_not_called()
                if mode in (RelevanceMode.SOFT, RelevanceMode.ENFORCE):
                    expected = "override" if mode == RelevanceMode.SOFT else "reject"
                    self.assertEqual(self.relevance.decision(query, "multimodal"), decision)
                    self.assertEqual(self.relevance.decision(query, "multimodal", True), expected)
                    self.assertEqual(self.relevance.decision(query, "visual", True), expected)

    def test_prompt_embeddings_are_cached(self):
        self.relevance.classifier = None
        self.provider.encode_text.return_value = np.ones(512)
        self.relevance.prepare("first.png")
        prompt_count = sum(
            len(prompts) for group in self.relevance.config["prompts"].values()
            for prompts in group.values()
        )
        self.assertEqual(self.provider.encode_text.call_count, prompt_count)
        self.relevance.prepare("second.png")
        self.assertEqual(self.provider.encode_text.call_count, prompt_count)

    def test_concurrent_initialization_encodes_each_prompt_once(self):
        self.relevance.classifier = None
        self.provider.encode_text.return_value = np.ones(512)
        barrier = Barrier(4)
        def evaluate():
            barrier.wait(timeout=5)
            return self.relevance.evaluate_embedding(np.ones(512))
        with ThreadPoolExecutor(max_workers=4) as threads:
            queries = [threads.submit(evaluate) for _ in range(4)]
            self.assertTrue(all(f.result().relevance.error is None for f in queries))
        self.assertEqual(self.provider.encode_text.call_count, 30)

    def test_initialization_failure_retries_with_same_embedding(self):
        self.relevance.classifier = None
        self.provider.encode_text.return_value = np.ones(512)
        embedding = self.relevance.encode_image("private.png")
        with patch.object(self.provider, "encode_text", side_effect=RuntimeError("private")):
            query = self.relevance.evaluate_embedding(embedding)
        self.assertEqual(query.relevance.error, "classifier_initialization_failed")
        self.assertIsNone(self.relevance.evaluate_embedding(embedding).relevance.error)
        self.provider.encode_image.assert_called_once()

    def test_classification_failure(self):
        self.relevance.classifier.score.side_effect = RuntimeError("private-path.png")
        for mode in (RelevanceMode.SHADOW, RelevanceMode.SOFT, RelevanceMode.ENFORCE):
            with self.subTest(mode=mode):
                self.relevance.mode = mode
                query = self.relevance.prepare("private-path.png")
                self.assertIsNotNone(query.relevance.error)
                self.assertEqual(self.relevance.decision(query, "multimodal"), "continue")
                self.assertEqual(self.relevance.decision(query, "visual"), "continue")


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        provider = Mock(spec=CLIPProvider)
        provider.MODEL_NAME = CLIPProvider.MODEL_NAME
        provider.encode_image.return_value = np.ones(512)
        self.engine = SearchEngine.__new__(SearchEngine)
        self.engine.vector = Mock()
        self.engine.vector.search_vector.return_value = [{"id": 2}, {"id": 1}]
        self.engine.vector.search_multimodal_vector.return_value = [{"id": 3}]
        self.engine.relevance = FashionRelevance(provider, RelevanceMode.SHADOW)
        self.engine.relevance.classifier = Mock()
        self.engine.relevance.classifier.score.return_value = NON_FASHION_SCORES
        self.stdout = io.StringIO()
        self.enterContext(patch("sys.stdout", self.stdout))
        self.enterContext(patch.object(logger, "handlers", []))
        self.enterContext(patch.object(logger, "level", logger.level))
        self.enterContext(patch.object(logger, "propagate", logger.propagate))
        configure_relevance_logging()
        configure_relevance_logging()

    def events(self):
        return [json.loads(line) for line in self.stdout.getvalue().splitlines()]

    def test_stdout_schema_and_deduplication(self):
        query = self.engine.relevance.prepare("private-upload.png")
        self.engine.search("multimodal", text="private query", image_query=query)
        self.engine.search("multimodal", text="private query", image_query=query, limit=5)
        event, = self.events()
        self.assertEqual(event["event"], "fashion_relevance_evaluation")
        self.assertEqual(UUID(event["query_id"]).version, 4)
        self.assertEqual(event["query_id"], query.query_id)
        self.assertEqual(event["search_mode"], "multimodal")
        self.assertEqual(event["relevance_mode"], "SHADOW")
        self.assertEqual(event["classification_status"], "classified")
        self.assertEqual(event["predicted_domain"], "non-fashion")
        self.assertEqual(event["category_scores"], {"clothing": 0.2, "logos": 0.3})
        self.assertEqual(event["score_margin"], -0.1)
        self.assertGreaterEqual(event["classification_latency_ms"], 0)
        self.assertGreaterEqual(event["search_latency_ms"], 0)
        self.assertEqual(event["result_count"], 1)
        self.assertNotEqual(event["application_version"], "unknown")
        self.assertTrue(event["classifier_version"].startswith("clip-zero-shot-v1:"))
        self.assertTrue(event["would_reject"])
        self.assertFalse(event["actually_rejected"])
        self.assertNotIn("private", self.stdout.getvalue())
        self.assertNotIn("embedding", self.stdout.getvalue())
        self.engine.search("visual", image_path="private-upload.png")
        self.assertNotEqual(self.events()[1]["query_id"], event["query_id"])

    def test_shadow_off_retrieval_parity(self):
        for mode in ("visual", "multimodal"):
            self.engine.relevance.mode = RelevanceMode.SHADOW
            shadow = self.engine.search(mode, text="bag", image_path="private.png")
            self.engine.relevance.mode = RelevanceMode.OFF
            self.assertEqual(shadow, self.engine.search(mode, text="bag", image_path="private.png"))
        self.assertEqual(len(self.events()), 2)
        self.assertEqual(self.engine.relevance.classifier.score.call_count, 2)

    def test_confirmation_and_enforcement(self):
        self.engine.relevance.mode = RelevanceMode.SOFT
        query = self.engine.relevance.prepare("private.png")
        with self.assertRaises(RelevanceBlocked):
            self.engine.search("visual", image_query=query)
        self.engine.search("visual", image_query=query, override_relevance=True)
        event, action = self.events()
        self.assertEqual(action["event"], "fashion_relevance_action")
        self.assertEqual(action["query_id"], event["query_id"])
        self.assertEqual(action["user_action"], "search_anyway")
        self.assertEqual(action["result_count"], 2)
        self.engine.search("visual", image_query=query, override_relevance=True, limit=5)
        self.assertEqual(len(self.events()), 2)
        self.assertEqual(event["search_status"], "awaiting_confirmation")
        self.assertTrue(event["would_reject"])
        self.assertFalse(event["actually_rejected"])
        self.assertIsNone(event["search_latency_ms"])
        self.assertIsNone(event["result_count"])
        self.engine.relevance.mode = RelevanceMode.ENFORCE
        with self.assertRaises(RelevanceBlocked):
            self.engine.search("visual", image_path="private.png")
        self.assertTrue(self.events()[2]["actually_rejected"])
        self.engine.vector.search_multimodal_vector.assert_not_called()
        with self.assertRaises(RelevanceBlocked):
            self.engine.search("multimodal", text="bag", image_path="private.png", override_relevance=True)
        self.engine.vector.search_multimodal_vector.assert_not_called()
        self.assertTrue(self.events()[3]["actually_rejected"])

    def test_classifier_failure_remains_distinct(self):
        self.engine.relevance.classifier.score.side_effect = RuntimeError("private data")
        self.assertEqual(len(self.engine.search("visual", image_path="private.png")), 2)
        event, = self.events()
        self.assertEqual(event["classification_status"], "failed")
        self.assertEqual(event["error_code"], "classifier_scoring_failed")
        self.assertEqual(event["model_version"], CLIPProvider.MODEL_NAME)
        self.assertEqual(len(event["prompt_version"]), 64)
        self.assertEqual(event["search_status"], "completed")
        self.assertIsNone(event["would_reject"])
        self.assertIsNone(event["category_scores"])
        self.assertFalse(event["actually_rejected"])
        self.assertNotIn("private", self.stdout.getvalue())

    def test_retrieval_failure_has_no_result_count(self):
        self.engine.vector.search_vector.side_effect = RuntimeError("private data")
        with self.assertRaises(RuntimeError):
            self.engine.search("visual", image_path="private.png")
        event, = self.events()
        self.assertEqual(event["classification_status"], "classified")
        self.assertEqual(event["search_status"], "failed")
        self.assertIsNone(event["result_count"])
        self.assertNotIn("private", self.stdout.getvalue())

    def test_logging_failure_does_not_interrupt_search(self):
        with patch.object(logger, "log", side_effect=OSError("sink unavailable")):
            self.assertEqual(len(self.engine.search("visual", image_path="private.png")), 2)

    def test_stdout_write_failure_stays_quiet(self):
        stderr = io.StringIO()
        with patch("sys.stderr", stderr), patch.object(self.stdout, "write", side_effect=OSError):
            self.assertEqual(len(self.engine.search("visual", image_path="private.png")), 2)
        self.assertEqual(stderr.getvalue(), "")


class DiscoveryEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.engine = Mock()
        provider = Mock(spec=CLIPProvider)
        provider.MODEL_NAME = CLIPProvider.MODEL_NAME
        provider.encode_image.side_effect = lambda path: np.ones(512)
        self.engine.relevance = FashionRelevance(provider, RelevanceMode.SOFT)
        self.engine.relevance.classifier = Mock()
        self.engine.relevance.classifier.score.return_value = NON_FASHION_SCORES
        self.engine.search.return_value = [{"id": 1}]
        self.state = Session()

    def request(self, image=b"first", text="shirt"):
        return {"mode": "Multimodal", "text": text, "image": image, "weight": 0.5}

    def search(self, request, state=None):
        with patch.object(discovery.st, "session_state", self.state if state is None else state):
            discovery.run_search(self.engine, request, "Discover", {}, 10)

    def test_text_edits_and_override_reuse_image_but_keep_distinct_query_ids(self):
        first, second = self.request(), self.request(text="bag")
        self.search(first)
        first["override_relevance"] = True
        self.search(first)
        self.search(second)
        self.engine.relevance.provider.encode_image.assert_called_once()
        self.assertIs(first["image_query"].embedding, second["image_query"].embedding)
        self.assertNotEqual(first["image_query"].query_id, second["image_query"].query_id)

    def test_replacement_removal_and_model_change_invalidate(self):
        first = self.request()
        self.search(first)
        self.state.discovery_request = first
        self.state["upload"] = object()
        with patch.object(discovery.st, "session_state", self.state):
            with patch.object(discovery, "prepare_image", return_value=b"second"):
                discovery.accept_upload("upload")
            self.assertNotIn("discovery_image_embedding", self.state)
            self.assertNotIn("discovery_request", self.state)
        second = self.request(image=b"second")
        self.search(second)
        self.state.discovery_request = second
        self.engine.relevance.model_version = "new-model"
        changed = second
        changed["override_relevance"] = True
        self.search(changed)
        self.assertNotIn("override_relevance", changed)
        self.state.discovery_request = changed
        with patch.object(discovery.st, "session_state", self.state):
            discovery.continue_without_image()
        self.assertNotIn("discovery_image_embedding", self.state)
        self.assertNotIn("image_query", changed)
        with patch.object(self.engine.relevance, "record_action") as record:
            self.search(changed)
            self.search(changed)
        record.assert_called_once()
        self.assertEqual(record.call_args.args[2:4], ("remove_image", "text_only_fallback"))
        self.assertEqual(self.engine.relevance.provider.encode_image.call_count, 3)

    def test_sessions_never_share_user_image_embeddings(self):
        first, second = self.request(), self.request()
        self.search(first)
        self.search(second, Session())
        self.assertEqual(self.engine.relevance.provider.encode_image.call_count, 2)
        self.assertIsNot(first["image_query"].embedding, second["image_query"].embedding)


class Session(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class OfflineSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
            raise RuntimeError("Run with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
        cls.engine = SearchEngine(RelevanceMode.SOFT)
        cls.product = next(
            p for p in cls.engine.catalog.matching_products("Discover", None, 20)
            if p["image_path"]
        )
        cls.negative = Path(
            "data/evaluation/clip_domain_independent/images/corporate-logos-hitachi-logo-220e14.png"
        )

    def test_all_six_modes_and_filter_parity(self):
        path = self.product["image_path"]
        for mode in ("keyword", "fuzzy", "semantic", "hybrid", "visual", "multimodal"):
            with self.subTest(mode=mode):
                options = {"text": "handbag", "limit": 3}
                if mode in ("visual", "multimodal"):
                    options["image_path"] = path
                    options["override_relevance"] = True
                self.assertTrue(self.engine.search(mode, **options))
        prepared = self.engine.relevance.prepare(path)
        eligible = {p["id"] for p in self.engine.catalog.matching_products("Bags", None)
                    if p["id"] != self.product["id"]}
        self.assertTrue(eligible)
        expected = self.engine.vector.search_vector(prepared.embedding, 5, eligible_ids=eligible)
        self.assertTrue(expected)
        self.assertEqual(expected, self.engine.search(
            "visual", image_query=prepared, override_relevance=True, collection="Bags",
            exclude_product_id=self.product["id"], limit=5,
        ))
        field = self.engine.catalog.filter_options()["colour"][0]
        filters = {"colour": field}
        filtered = self.engine.search(
            "visual", image_query=prepared, override_relevance=True, filters=filters,
        )
        self.assertTrue(filtered)
        self.assertTrue(all(p["colour"] == field for p in filtered))

    def test_fashion_and_classification_cost(self):
        image = Path("data/sample_product.jpg")
        provider = self.engine.vector.provider
        expected = self.engine.vector.search_image(image, 5)
        with patch.object(provider, "encode_image", wraps=provider.encode_image) as encode:
            prepared = self.engine.relevance.prepare(image)
            self.assertEqual(prepared.relevance.domain, "fashion")
            self.assertEqual(expected, self.engine.search("visual", image_query=prepared, limit=5))
            encode.assert_called_once()
        times = []
        for _ in range(200):
            start = perf_counter()
            self.engine.relevance.classifier.score(prepared.embedding)
            times.append((perf_counter() - start) * 1000)
        print(f"Additional zero-shot scoring over 200 repeats median {median(times):.3f} ms "
              f"and P95 {sorted(times)[189]:.3f} ms")

    def test_shadow_preserves_real_filtered_rankings(self):
        path = self.negative if self.negative.is_file() else Path("data/sample_product.jpg")
        for mode in ("visual", "multimodal"):
            options = dict(image_path=path, limit=5, collection="Bags", text="black handbag")
            with self.subTest(mode=mode):
                with patch.object(self.engine.relevance, "mode", RelevanceMode.OFF):
                    expected = self.engine.search(mode, **options)
                with patch.object(self.engine.relevance, "mode", RelevanceMode.SHADOW):
                    self.assertTrue(expected)
                    self.assertEqual(expected, self.engine.search(mode, **options))

    def test_engine_remains_available_when_classifier_fails(self):
        path = Path("data/sample_product.jpg")
        expected = self.engine.vector.search_image(path, 5)
        self.engine.relevance.prepare(path)
        with patch.object(self.engine.relevance.classifier, "score", side_effect=RuntimeError):
            for mode in (RelevanceMode.SHADOW, RelevanceMode.SOFT, RelevanceMode.ENFORCE):
                with self.subTest(mode=mode), patch.object(self.engine.relevance, "mode", mode):
                    with self.assertLogs("app.search.relevance", "WARNING") as logs:
                        self.assertEqual(expected, self.engine.search("visual", image_path=path, limit=5))
                        multimodal = self.engine.search("multimodal", text="handbag", image_path=path, limit=5)
                    self.assertTrue(multimodal)
                    events = [json.loads(record.getMessage()) for record in logs.records]
                    self.assertEqual([e["search_mode"] for e in events], ["visual", "multimodal"])
                    self.assertTrue(all(e["search_status"] == "completed" and not e["actually_rejected"]
                                        for e in events))

    def test_generated_non_fashion_and_multimodal_single_encoding(self):
        report = Image.new("RGB", (320, 380), "white")
        draw = ImageDraw.Draw(report)
        draw.text((24, 22), "ANNUAL REPORT", font=ImageFont.load_default(size=30), fill="black")
        draw.text((24, 67), "FINANCIAL STATEMENTS", font=ImageFont.load_default(size=18), fill="black")
        for y in (112, 146, 180, 214, 248, 282, 316):
            draw.line((24, y, 296, y), fill="#444444", width=2)
            draw.text((34, y + 7), "Operating expenses         21,609", fill="black")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "report.png"
            report.save(path)
            self.check_non_fashion_search(path)

    def test_independent_logo(self):
        if not self.negative.is_file():
            self.skipTest("Local Wikimedia evaluation logo is excluded from Git and Docker")
        self.check_non_fashion_search(self.negative)

    def check_non_fashion_search(self, path):
        provider = self.engine.vector.provider
        with patch.object(provider, "encode_image", wraps=provider.encode_image) as encode:
            prepared = self.engine.relevance.prepare(path)
            self.assertEqual(prepared.relevance.domain, "non-fashion")
            with self.assertRaises(RelevanceBlocked):
                self.engine.search("visual", image_query=prepared)
            self.assertTrue(self.engine.search("visual", image_query=prepared, override_relevance=True))
            expected = self.engine.vector.search_multimodal_vector("handbag", prepared.embedding, 0.3, 10)
            self.assertTrue(expected)
            self.assertEqual(expected, self.engine.search(
                "multimodal", text="handbag", image_query=prepared, image_weight=0.3,
                override_relevance=True,
            ))
            encode.assert_called_once()
        with patch.object(provider, "encode_image", wraps=provider.encode_image) as encode:
            with self.assertRaises(RelevanceBlocked):
                self.engine.search("multimodal", text="handbag", image_path=path)
            encode.assert_called_once()
        self.assertTrue(self.engine.search("semantic", text="handbag"))


if __name__ == "__main__":
    unittest.main()
