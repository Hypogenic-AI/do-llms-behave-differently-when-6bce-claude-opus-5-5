Project: Probing and steering evaluation awareness. Training probes on llama 3.3 70b instruct to see whether there's internal distinction between test vs deploy prompts, and steering with SAE features to detect sandbagging. This repo is a minimal implementation of probe training and evaluation.

Paper: https://arxiv.org/abs/2507.01786

Results: Linear probes trained with simple contrastive data generalises to distinguishing real eval and deploy prompts. Steering with this direction has no causal effects, but with SAE features has some effects on showing sandbagging.




**Usage**

1. Clone the repo
2. Get packages ```pip install -r requirements.txt```
3. Generate probes ```python scripts/generate_vectors.py --model MODEL_PATH --data DATASET_PATH --output OUTPUT_DIR```
4. Visualise ```python scripts/analyze_probe.py --model MODEL_PATH --vectors VECTORS_DIR --data TEST_DATA --output RESULTS_DIR --visualize```
