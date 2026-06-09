"""
src/topic_registry/registry.py
-------------------------------
TopicRegistry: source-of-truth for all topic IDs, names, statuses,
and metadata. Append-only — IDs are never reassigned or deleted.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

VALID_STATUSES = {"active", "deprecated", "merged_into"}


class TopicRegistry:
    """
    Manages the persistent topic registry JSON file.
    All mutations go through this class — never edit the JSON directly.
    """

    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self._data: dict = {}
        self.load()

    # ── Load / Save ───────────────────────────────────────────────────────────

    def load(self) -> None:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            log.info("Registry loaded: %d topics.", len(self._data.get("topics", {})))
        else:
            self._data = {
                "schema_version": "1.0",
                "last_updated": self._now(),
                "total_topics": 0,
                "topics": {},
            }
            log.info("No registry found — initialised empty registry.")

    def save(self) -> None:
        self._data["last_updated"] = self._now()
        self._data["total_topics"] = len(self._data["topics"])
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        log.info("Registry saved: %d topics.", self._data["total_topics"])

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_topic(self, topic_id: int) -> Optional[dict]:
        return self._data["topics"].get(str(topic_id))

    def get_name(self, topic_id: int) -> str:
        topic = self.get_topic(topic_id)
        if topic is None:
            return "Uncategorised"
        if topic["status"] == "merged_into":
            return self.get_name(topic["merged_into_id"])
        if topic["status"] == "deprecated":
            return "Uncategorised"
        return topic["name"]

    def get_active_topic_ids(self) -> list[int]:
        """Returns IDs of all active topics (not deprecated, not merged)."""
        return [
            int(tid)
            for tid, meta in self._data["topics"].items()
            if meta["status"] == "active"
        ]

    def next_available_id(self) -> int:
        """Returns the next integer ID that has never been used."""
        if not self._data["topics"]:
            return 0
        return max(int(k) for k in self._data["topics"].keys()) + 1

    def is_empty(self) -> bool:
        return len(self._data["topics"]) == 0

    def topic_count(self) -> int:
        return len(self._data["topics"])

    def active_topic_count(self) -> int:
        return len(self.get_active_topic_ids())

    # ── Write (append-only) ───────────────────────────────────────────────────

    def add_topic(
        self,
        topic_id: int,
        name: str,
        model_version: str,
        discovery_run_id: str,
        keywords: list[str],
        representative_doc_ids: list[str],
        review_count_at_creation: int,
        notes: str = "",
    ) -> None:
        key = str(topic_id)
        if key in self._data["topics"]:
            raise ValueError(
                f"Topic ID {topic_id} already exists in registry. "
                "IDs are immutable — use next_available_id()."
            )
        self._data["topics"][key] = {
            "name": name,
            "status": "active",
            "model_version": model_version,
            "discovery_run_id": discovery_run_id,
            "created_at": self._now(),
            "deprecated_at": None,
            "merged_into_id": None,
            "keyword_signature": keywords,
            "representative_doc_ids": representative_doc_ids,
            "review_count_at_creation": review_count_at_creation,
            "notes": notes,
        }
        log.info("Topic %d registered: '%s' [%s]", topic_id, name, model_version)

    def deprecate_topic(self, topic_id: int, reason: str = "") -> None:
        key = str(topic_id)
        if key not in self._data["topics"]:
            raise KeyError(f"Topic ID {topic_id} not found in registry.")
        self._data["topics"][key]["status"] = "deprecated"
        self._data["topics"][key]["deprecated_at"] = self._now()
        self._data["topics"][key]["notes"] += f" | Deprecated: {reason}"
        log.info("Topic %d deprecated. Reason: %s", topic_id, reason)

    def merge_topics(self, source_id: int, target_id: int) -> None:
        """
        Marks source_id as merged into target_id.
        Future assignments to source_id are transparently re-routed to target_id.
        """
        source_key = str(source_id)
        target_key = str(target_id)
        if source_key not in self._data["topics"]:
            raise KeyError(f"Source topic {source_id} not in registry.")
        if target_key not in self._data["topics"]:
            raise KeyError(f"Target topic {target_id} not in registry.")
        self._data["topics"][source_key]["status"] = "merged_into"
        self._data["topics"][source_key]["merged_into_id"] = target_id
        self._data["topics"][source_key]["deprecated_at"] = self._now()
        log.info("Topic %d merged into topic %d.", source_id, target_id)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def summary(self) -> dict:
        statuses = {}
        for meta in self._data["topics"].values():
            statuses[meta["status"]] = statuses.get(meta["status"], 0) + 1
        return {
            "total_topics": self.topic_count(),
            "active": statuses.get("active", 0),
            "deprecated": statuses.get("deprecated", 0),
            "merged_into": statuses.get("merged_into", 0),
        }