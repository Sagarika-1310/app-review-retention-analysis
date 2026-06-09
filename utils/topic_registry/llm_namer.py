"""
src/topic_registry/llm_namer.py
---------------------------------
LLMNamer: generates human-readable cluster names via locally running
Llama 3 through Ollama. Fully offline, zero cost, no data egress.

Fallback: top-3 BERTopic keywords joined if Ollama is unreachable.
"""

import logging
import requests

log = logging.getLogger(__name__)

DEFAULT_OLLAMA_URL   = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3"


class LLMNamer:

    def __init__(
        self,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        ollama_model: str = DEFAULT_OLLAMA_MODEL,
        timeout_seconds: int = 60,
    ):
        self.ollama_url   = ollama_url
        self.ollama_model = ollama_model
        self.timeout      = timeout_seconds
        self._ollama_available: bool | None = None   # cached after first probe

    def name_cluster(self, keywords: list[str], sample_reviews: list[str]) -> str:
        """
        Returns a 2–4 word pain point category name for a cluster.
        Uses Llama 3 if Ollama is reachable, otherwise keyword fallback.
        """
        if not self._is_ollama_available():
            return self._keyword_fallback(keywords)

        prompt = self._build_prompt(keywords, sample_reviews)
        try:
            response = requests.post(
                self.ollama_url,
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw  = response.json().get("response", "").strip()
            name = raw.strip("\"'.,\n ").strip()
            if not name or len(name) > 60:
                return self._keyword_fallback(keywords)
            return name
        except requests.exceptions.Timeout:
            log.warning("Ollama timed out after %ds. Using keyword fallback.", self.timeout)
            return self._keyword_fallback(keywords)
        except Exception as e:
            log.warning("LLM naming failed: %s. Using keyword fallback.", e)
            return self._keyword_fallback(keywords)

    def name_all_clusters(
        self,
        cluster_data: dict[int, dict],
    ) -> dict[int, str]:
        """
        Names all clusters in a discovery run.
        {raw_cluster_id: {"keywords": [...], "texts": [...]}} → {raw_cluster_id: name}
        """
        names = {}
        for cluster_id, data in cluster_data.items():
            name = self.name_cluster(
                keywords=data.get("keywords", []),
                sample_reviews=data.get("texts", [])[:5],
            )
            names[cluster_id] = name
            log.info("Cluster %d → '%s'", cluster_id, name)
        return names

    # ── Private ───────────────────────────────────────────────────────────────

    def _is_ollama_available(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available

        try:
            r = requests.get("http://localhost:11434/", timeout=3)
            self._ollama_available = r.status_code == 200
        except Exception:
            self._ollama_available = False
            log.warning(
                "Ollama not reachable at %s. "
                "All cluster names will use keyword fallback. "
                "Start Ollama with: ollama serve && ollama pull %s",
                self.ollama_url,
                self.ollama_model,
            )
        return self._ollama_available

    @staticmethod
    def _build_prompt(keywords: list[str], sample_reviews: list[str]) -> str:
        kw_str  = ", ".join(keywords[:15])
        rev_str = "\n".join(f"- {r[:200]}" for r in sample_reviews[:5])
        return (
            "You are a product analyst specialising in customer feedback. "
            "Based on the BERTopic keywords and sample reviews below, "
            "generate a concise 2–4 word pain point category name that precisely "
            "describes the user complaint theme. "
            "Output ONLY the category name. No explanation. No punctuation. No quotes.\n\n"
            f"Keywords: {kw_str}\n\n"
            f"Sample reviews:\n{rev_str}\n\n"
            "Pain point category name:"
        )

    @staticmethod
    def _keyword_fallback(keywords: list[str]) -> str:
        if not keywords:
            return "Unknown Issue"
        clean = [k.strip().title() for k in keywords[:3] if k.strip()]
        return " / ".join(clean) if clean else "Unknown Issue"