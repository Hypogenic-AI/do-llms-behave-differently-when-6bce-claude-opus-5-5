"""Validity check of the verbal self-report read-out: can each local model tell human from
LLM text when asked, on (a) HC3 answers (human vs ChatGPT; documents, not prompts) and (b)
synthetic_polistance prompts (real users vs LLM-generated)? Reports AUROC of the log-odds."""
import json
import os
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

from common import DATA, RES, SEED, read_jsonl
from local_lm import LM

key = sys.argv[1]
lm = LM(key)
hc = pd.DataFrame(read_jsonl(os.path.join(DATA, "HC3/all.jsonl"))).sample(200, random_state=SEED)
hum = [h[0][:800] for h in hc.human_answers if h]
gpt = [g[0][:800] for g in hc.chatgpt_answers if g]
sp = pd.read_csv(os.path.join(DATA, "synthetic_polistance/prompts.csv"))
real = sp[sp.arm == "real"].prompt.sample(200, random_state=SEED).tolist()
gen = sp[sp.arm == "llm_generated"].prompt.sample(200, random_state=SEED).tolist()
out = {}
for name, pos, neg in [("hc3_answers", gpt, hum), ("polistance_prompts", gen, real)]:
    s_pos, s_neg = lm.p_ai(pos), lm.p_ai(neg)
    out[name] = {"auroc": roc_auc_score([1] * len(pos) + [0] * len(neg), s_pos + s_neg),
                 "mean_logodds_llm": sum(s_pos) / len(s_pos), "mean_logodds_human": sum(s_neg) / len(s_neg)}
print(key, out)
json.dump(out, open(os.path.join(RES, f"selfreport_validity_{key}.json"), "w"), indent=1)
