"""Experiment 1 (API models): responses for every item x condition via OpenRouter (cached)."""
import argparse
import os

from common import API_RESPONDERS, RES, chat, pmap, write_jsonl
from conditions import jobs, load_items, max_tokens, messages

ap = argparse.ArgumentParser()
ap.add_argument("--models", default=",".join(API_RESPONDERS))
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()

items = load_items()[: a.limit]
J = jobs(items)
for m in a.models.split(","):
    def f(job):
        it, c = job
        # Gemini spends part of max_tokens on (unremovable, low-effort) reasoning: give it headroom.
        mt = max_tokens(it) + (600 if "gemini" in m else 0)
        r = chat(m, messages(it, c), max_tokens=mt, temperature=0.0)
        return dict(id=it["id"], task=it["task"], cond=c, model=m.split("/")[1], response=r["text"],
                    error=r.get("error"), finish=r.get("finish"), tok_in=r["in"], tok_out=r["out"])
    out = pmap(f, J, workers=32, desc=m)
    short = m.split("/")[1]
    write_jsonl(os.path.join(RES, f"responses_{short}.jsonl"), out)
    n_err = sum(o["response"] is None for o in out)
    print(m, len(out), "errors", n_err, "tokens in/out", sum(o["tok_in"] for o in out), sum(o["tok_out"] for o in out))
