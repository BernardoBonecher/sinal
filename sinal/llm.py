"""Minimal client for a local OpenAI-compatible server (LM Studio by default).

- Structured output: every call sends a JSON schema (`response_format: json_schema`).
- Cache: responses are stored by a hash of (model, messages, schema), so a run can be replayed,
  and the report rebuilt, without the model.
- Log: one JSON line per call with model, latency and token counts (a small, local stand-in for
  an observability tool).

Standard library only.
"""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = os.environ.get("SINAL_LLM_URL", "http://localhost:1234/v1")


class LLMError(RuntimeError):
    pass


class LocalLLM:
    def __init__(self, model: str, run_dir: Path, base_url: str = DEFAULT_URL,
                 offline: bool = False, timeout: int = 1800):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.offline = offline
        self.timeout = timeout
        self.cache_dir = Path(run_dir) / "cache"
        self.log_path = Path(run_dir) / "calls.jsonl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, messages: list[dict], schema: dict, temperature: float) -> str:
        blob = json.dumps([self.model, messages, schema, temperature], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as err:
            raise LLMError(f"Could not reach the model server at {self.base_url} ({err}). "
                           "Is LM Studio's server running with a model loaded?") from err

    def json(self, name: str, messages: list[dict], schema: dict, temperature: float = 0.0,
             label: str = "", max_tokens: int | None = None) -> dict:
        """Run one structured call and return the parsed JSON object."""
        key = self._key(messages, schema, temperature)
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))["parsed"]
        if self.offline:
            raise LLMError(f"Offline mode and no cached response for call '{label or name}' ({key}).")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}},
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        start = time.perf_counter()
        body = self._post(payload)
        seconds = time.perf_counter() - start
        choice = body["choices"][0]
        content = choice["message"]["content"]
        if choice.get("finish_reason") == "length":
            raise LLMError(f"Model hit the token limit on '{label or name}' (likely a repetition loop); "
                           "nothing was cached. Try again or use a larger model.")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as err:
            raise LLMError(f"Model returned invalid JSON for '{label or name}': {content[:200]}") from err

        usage = body.get("usage", {})
        record = {
            "key": key, "label": label or name, "model": body.get("model", self.model),
            "seconds": round(seconds, 2),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
        }
        with open(self.log_path, "a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
        cached.write_text(json.dumps({"record": record, "parsed": parsed}, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        return parsed
