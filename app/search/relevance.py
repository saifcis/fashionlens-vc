"""Fashion-domain diagnostics and policy for image queries."""

import hashlib
import json
import logging
import os
import sys
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

import numpy as np

from app.search.clip_domain import ClipDomainClassifier


logger = logging.getLogger(__name__)
_logging_lock = Lock()


class _StdoutEventHandler(logging.StreamHandler):
    def handleError(self, record):
        # A broken stdout must not dump a logging traceback to stderr.
        pass


def configure_relevance_logging():
    with _logging_lock:
        if not logger.handlers:
            handler = _StdoutEventHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False


def application_version():
    project_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        with project_path.open("rb") as project:
            return tomllib.load(project)["project"]["version"]
    except (OSError, KeyError, ValueError):
        return "unknown"


class RelevanceMode(str, Enum):
    OFF = "OFF"
    SHADOW = "SHADOW"
    SOFT = "SOFT"
    ENFORCE = "ENFORCE"

    @classmethod
    def configured(cls):
        return cls(os.getenv("FASHION_LENS_RELEVANCE_MODE", "SOFT").strip().upper())


@dataclass(frozen=True)
class RelevanceDiagnostic:
    domain: str | None
    scores: dict | None
    latency_ms: float | None
    error: str | None = None


@dataclass(eq=False)
class ImageQuery:
    embedding: np.ndarray
    relevance: RelevanceDiagnostic
    query_id: str = field(default_factory=lambda: uuid4().hex, init=False)
    _event_claimed: bool = field(default=False, init=False, repr=False)
    _override_claimed: bool = field(default=False, init=False, repr=False)


class RelevanceBlocked(ValueError):
    def __init__(self, decision, diagnostic):
        super().__init__("The image did not pass fashion relevance review.")
        self.decision = decision
        self.diagnostic = diagnostic


class FashionRelevance:
    def __init__(self, provider, mode):
        self.provider = provider
        self.mode = mode
        self.classifier = None
        self.lock = Lock()
        config_bytes = Path(__file__).with_name("fashion_prompts.json").read_bytes()
        self.config = json.loads(config_bytes)
        self.prompt_version = hashlib.sha256(config_bytes).hexdigest()
        self.classifier_version = f"clip-zero-shot-v1:{self.prompt_version}"
        model_config = getattr(getattr(provider, "model", None), "config", None)
        revision = getattr(model_config, "_commit_hash", None)
        self.model_version = provider.MODEL_NAME
        if isinstance(revision, str) and revision:
            self.model_version = f"{self.model_version}@{revision}"
        self.application_version = application_version()
        self.application_revision = os.getenv("K_REVISION")

    @property
    def configuration(self):
        return self.model_version, self.classifier_version, self.mode

    def encode_image(self, image_path):
        embedding = self.provider.encode_image(image_path)
        embedding.setflags(write=False)
        return embedding

    def prepare(self, image_path):
        return self.evaluate_embedding(self.encode_image(image_path))

    def evaluate_embedding(self, embedding):
        if self.mode == RelevanceMode.OFF:
            return ImageQuery(embedding, RelevanceDiagnostic(None, None, None))

        start = None
        try:
            if self.classifier is None:
                with self.lock:
                    if self.classifier is None:
                        if self.provider.MODEL_NAME != self.config["model"]:
                            raise ValueError("Fashion relevance model mismatch")
                        self.classifier = ClipDomainClassifier(self.provider, self.config)
            start = perf_counter()
            scores = self.classifier.score(embedding)
            latency = (perf_counter() - start) * 1000
            cosines = (*scores["category_scores"].values(), scores["margin"])
            if not all(np.isfinite(value) for value in cosines):
                raise ValueError("Invalid fashion relevance scores")
            domain = "fashion" if scores["margin"] >= self.config["zero_shot_margin"] else "non-fashion"
            return ImageQuery(embedding, RelevanceDiagnostic(domain, scores, latency))
        except Exception:
            # Exception text can contain request data or paths supplied by callers.
            latency = (perf_counter() - start) * 1000 if start is not None else None
            code = "classifier_scoring_failed" if start is not None else "classifier_initialization_failed"
            return ImageQuery(embedding, RelevanceDiagnostic(None, None, latency, code))

    def decision(self, query, search_mode, override=False):
        if self.mode in (RelevanceMode.OFF, RelevanceMode.SHADOW):
            return "continue"
        if query.relevance.domain != "non-fashion" or query.relevance.error:
            return "continue"
        if self.mode == RelevanceMode.ENFORCE:
            return "reject"
        return "override" if override else "confirm"

    def _context(self, query_id, search_mode):
        return {
            "query_id": query_id,
            "search_mode": search_mode,
            "relevance_mode": self.mode.value,
            "application_version": self.application_version,
            "application_revision": self.application_revision,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "classifier_version": self.classifier_version,
        }

    def record_evaluation(self, query, search_mode, decision, *, search_status,
                          search_latency_ms=None, result_count=None):
        if self.mode == RelevanceMode.OFF:
            return
        with self.lock:
            if query._event_claimed:
                return
            query._event_claimed = True

        # Diagnostics are best effort and never affect retrieval or policy decisions.
        try:
            diagnostic = query.relevance
            scores = diagnostic.scores or {}
            failed = diagnostic.error is not None
            payload = {
                "event": "fashion_relevance_evaluation",
                "severity": "WARNING" if failed or search_status == "failed" else "INFO",
                **self._context(query.query_id, search_mode),
                "user_action": "search_anyway" if decision == "override" else "submit",
                "classification_status": "failed" if failed else "classified",
                "error_code": diagnostic.error,
                "predicted_domain": diagnostic.domain,
                "fashion_category": scores.get("fashion_category"),
                "non_fashion_category": scores.get("non_fashion_category"),
                "category_scores": scores.get("category_scores"),
                "score_margin": scores.get("margin"),
                "classification_latency_ms": diagnostic.latency_ms,
                "search_status": search_status,
                "search_latency_ms": search_latency_ms,
                "result_count": result_count,
                "decision": decision,
                "would_reject": None if failed else diagnostic.domain == "non-fashion",
                "actually_rejected": decision == "reject",
            }
            self._emit(payload)
        except Exception:
            # A failing sink may have written already. Avoid retrying the event.
            pass

    def record_override(self, query, search_mode, **execution):
        with self.lock:
            if query._override_claimed:
                return
            query._override_claimed = True
        self.record_action(query.query_id, search_mode, "search_anyway", "override", **execution)

    def record_action(self, query_id, search_mode, action, decision, *, search_status,
                      search_latency_ms, result_count):
        self._emit({
            "event": "fashion_relevance_action",
            "severity": "WARNING" if search_status == "failed" else "INFO",
            **self._context(query_id, search_mode),
            "user_action": action,
            "decision": decision,
            "search_status": search_status,
            "search_latency_ms": search_latency_ms,
            "result_count": result_count,
        })

    @staticmethod
    def _emit(payload):
        try:
            level = logging.WARNING if payload["severity"] == "WARNING" else logging.INFO
            logger.log(level, json.dumps(payload, allow_nan=False, separators=(",", ":")))
        except Exception:
            pass
