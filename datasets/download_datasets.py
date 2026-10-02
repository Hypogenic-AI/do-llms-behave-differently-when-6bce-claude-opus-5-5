"""Re-download every dataset used by this project into datasets/.

Run from the workspace root:  python datasets/download_datasets.py
Needs: huggingface_hub, pandas, pyarrow, pytz (all in pyproject.toml).
"""
import os

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = os.path.dirname(os.path.abspath(__file__))

# (HF repo id, local dir name, allow_patterns or None for everything)
SNAPSHOTS = [
    ("JailbreakBench/JBB-Behaviors", "JBB-Behaviors", ["data/*", "README.md", "LICENSE"]),
    ("Anthropic/model-written-evals", "model-written-evals", None),
    ("viliana-dev/eval-awareness-2x2", "eval-awareness-2x2", None),
    ("Hello-SimpleAI/HC3", "HC3", ["all.jsonl", "README.md"]),
    ("truthfulqa/truthful_qa", "truthful_qa", None),
    ("meg-tong/sycophancy-eval", "sycophancy-eval", None),
    ("Paul/XSTest", "XSTest", None),
    ("kiddothe2b/synthetic_polistance", "synthetic_polistance", None),
]

# WildChat-1M is 3.4 GB in 14 shards; two shards (~120k conversations) are enough here.
WILDCHAT_SHARDS = (0, 1)


def build_wildchat_subset(wc_dir):
    """English, first user turn, 20-1500 chars, de-duplicated."""
    frames = [
        pd.read_parquet(os.path.join(wc_dir, f"data/train-{i:05d}-of-00014.parquet"))
        for i in WILDCHAT_SHARDS
    ]
    w = pd.concat(frames, ignore_index=True)
    e = w[(w.language == "English") & (~w.toxic)].copy()
    e["first_user"] = e.conversation.apply(
        lambda c: c[0]["content"] if len(c) and c[0]["role"] == "user" else None
    )
    e["first_assistant"] = e.conversation.apply(
        lambda c: c[1]["content"] if len(c) > 1 and c[1]["role"] == "assistant" else None
    )
    e = e.dropna(subset=["first_user"])
    e["n_chars"] = e.first_user.str.len()
    e = e[(e.n_chars >= 20) & (e.n_chars <= 1500)].drop_duplicates("first_user")
    out = e[
        ["conversation_hash", "model", "timestamp", "turn", "first_user",
         "first_assistant", "n_chars", "country"]
    ].copy()
    out["timestamp"] = out.timestamp.astype(str)
    out.to_parquet(os.path.join(wc_dir, "wildchat_en_first_turn_subset.parquet"), index=False)
    return len(out)


def main():
    for repo, name, patterns in SNAPSHOTS:
        snapshot_download(
            repo, repo_type="dataset", local_dir=os.path.join(ROOT, name), allow_patterns=patterns
        )
        print("downloaded", repo)

    wc_dir = os.path.join(ROOT, "WildChat-1M")
    for i in WILDCHAT_SHARDS:
        hf_hub_download(
            "allenai/WildChat-1M", f"data/train-{i:05d}-of-00014.parquet",
            repo_type="dataset", local_dir=wc_dir,
        )
    hf_hub_download("allenai/WildChat-1M", "README.md", repo_type="dataset", local_dir=wc_dir)
    print("WildChat English first-turn subset rows:", build_wildchat_subset(wc_dir))

    # jjpn2/eval_awareness is gated: accept the access conditions on its HF page with the
    # account behind HF_TOKEN, then uncomment. The data is an encrypted zip (scripts/decrypt.sh).
    # snapshot_download("jjpn2/eval_awareness", repo_type="dataset",
    #                   local_dir=os.path.join(ROOT, "eval_awareness"))


if __name__ == "__main__":
    main()
