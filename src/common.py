"""Shared utilities: paths, seeding, a cached + retrying OpenRouter client, parallel map."""
import hashlib
import json
import os
import random
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

SEED = 42

# Model roles (chosen 2026-10-02 from the live OpenRouter catalog; see planning.md)
REWRITERS = {"A": "x-ai/grok-4.3", "B": "deepseek/deepseek-v4-flash"}
API_RESPONDERS = ["openai/gpt-5.6-luna", "google/gemini-3.5-flash-lite"]
# Equivalence judge for rewrites (gpt-5.4-mini was tested first: it refused / hit content filters
# on XSTest/JBB items, so a non-OpenAI judge is used).
JUDGE = "anthropic/claude-sonnet-5"
# Response judge (refusal / answer grading): cheap, validated against JBB human labels.
RESP_JUDGE = "deepseek/deepseek-v4-pro"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class Cache:
    """sqlite key-value cache so every API call is made once and is reproducible offline."""

    def __init__(self, path=os.path.join(RES, "api_cache.sqlite")):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=120)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.db.commit()

    def get(self, k):
        with self.lock:
            r = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, k, v):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, json.dumps(v)))
            self.db.commit()


_cache = Cache()
_client = openai.OpenAI(api_key=os.environ["OPENROUTER_KEY"], base_url="https://openrouter.ai/api/v1",
                        timeout=120)


def _reasoning_cfg(model):
    # Gemini flash-lite cannot disable reasoning; use the lowest effort it accepts.
    return {"effort": "low"} if "gemini" in model else {"effort": "none"}


@retry(wait=wait_exponential(min=2, max=60), stop=stop_after_attempt(6))
def _call(model, messages, max_tokens, temperature):
    r = _client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
        extra_body={"reasoning": _reasoning_cfg(model)})
    if not r.choices:
        raise RuntimeError(f"no choices: {r}")
    content = r.choices[0].message.content
    if content is None:
        content = ""
    u = r.usage
    return {"text": content, "in": u.prompt_tokens if u else 0, "out": u.completion_tokens if u else 0,
            "finish": r.choices[0].finish_reason}


def chat(model, messages, max_tokens=300, temperature=0.0, tag=""):
    """Cached chat completion. `tag` distinguishes intentionally repeated samples."""
    key = hashlib.sha256(json.dumps([model, messages, max_tokens, temperature, tag]).encode()).hexdigest()
    hit = _cache.get(key)
    if hit is not None:
        return hit
    try:
        out = _call(model, messages, max_tokens, temperature)
    except Exception as e:  # keep going; record the failure explicitly
        return {"text": None, "error": str(e)[:300], "in": 0, "out": 0}
    _cache.put(key, out)
    return out


def pmap(fn, items, workers=24, desc=""):
    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(fn, items), total=len(items), desc=desc))
