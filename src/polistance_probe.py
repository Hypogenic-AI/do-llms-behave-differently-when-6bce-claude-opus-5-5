"""Is there a provenance signal beyond style? synthetic_polistance has real user prompts and
LLM-generated prompts written to imitate casual users (same topics/intents). We train a probe
*in-domain* (5-fold CV) on the local model's activations and compare it with two surface
baselines that never see the model: prompt length, and a TF-IDF bag-of-n-grams classifier.
If activations beat the text baselines, the model holds provenance information beyond surface form."""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, RES, SEED
from local_lm import LM

key = sys.argv[1]
sp = pd.read_csv(os.path.join(DATA, "synthetic_polistance/prompts.csv"))
real = sp[sp.arm == "real"]
gen = sp[sp.arm == "llm_generated"].sample(len(real), random_state=SEED)
df = pd.concat([real, gen])
texts, y = df.prompt.tolist(), (df.arm == "llm_generated").astype(int).values
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
out = {"n": len(df)}
out["length_auroc"] = roc_auc_score(y, [len(t) for t in texts])
tf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), LogisticRegression(C=1, max_iter=2000))
out["tfidf_cv_auroc"] = roc_auc_score(y, cross_val_predict(tf, texts, y, cv=cv, method="predict_proba")[:, 1])
tfc = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True),
                    LogisticRegression(C=1, max_iter=2000))
out["char_tfidf_cv_auroc"] = roc_auc_score(y, cross_val_predict(tfc, texts, y, cv=cv, method="predict_proba")[:, 1])

lm = LM(key)
X = lm.resid_last([[{"role": "user", "content": t}] for t in texts]).numpy()
d_style = __import__("torch").load(os.path.join(RES, f"dirs_{key}.pt"))["d"].numpy()
out["layers"] = {}
for l in range(2, X.shape[1], 2):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000))
    p = cross_val_predict(clf, X[:, l], y, cv=cv, method="predict_proba")[:, 1]
    w = clf.fit(X[:, l], y)[-1].coef_[0] / clf[0].scale_  # direction in raw activation space
    cos = float(w @ d_style[l] / (np.linalg.norm(w) * np.linalg.norm(d_style[l])))
    out["layers"][l] = {"probe_cv_auroc": roc_auc_score(y, p), "cos_with_style_dir": cos}
best = max(out["layers"], key=lambda l: out["layers"][l]["probe_cv_auroc"])
out["best_layer"] = best
print(key, {k: v for k, v in out.items() if k != "layers"}, out["layers"][best])
json.dump(out, open(os.path.join(RES, f"polistance_probe_{key}.json"), "w"), indent=1)
