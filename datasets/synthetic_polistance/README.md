---
license: cc-by-nc-sa-4.0
configs:
- config_name: prompts
  data_files: prompts.csv
- config_name: realness_ranking
  data_files: realness_ranking.csv
- config_name: detection_annotations
  data_files: detection_annotations.csv
- config_name: topics_details
  data_files: topics_details.csv
task_categories:
- text-generation
language:
- en
tags:
- politics
- stance
size_categories:
- 10K<n<100K
---

# Fully Synthetic Prompts for LLM Political Stance Detection

All resources developed in the article "*Templated or fully Synthetic? Prompt construction as a confound in measuring LLM political stance beyond writing assistance*" ([Chalkidis, 2026](https://arxiv.org/abs/2608.11008)).

## Paper Abstract

Political stance detection in LLMs has long been dominated by closed-ended, multiple-choice political survey questions—originally designed for humans, and thus lacks the realism and nuance of human-AI interactions in the wild, while also being susceptible to sandbagging. The recent IssueBench framework substantially mitigates these limitations with templated prompts anchored in real-world chat logs. 

Given the rise in non-work-related use of GenAI assistants, we extend IssueBench beyond writing assistance to include two additional tasks, information seeking and opinion sharing. We argue that templated prompts still lack the nuance of real ones, especially for open-ended tasks, and remain recognisable as evaluation artefacts. We propose the use of fully synthetic (LLM-generated) prompts, produced under detailed instructions with real prompts as seeds. We assess the ecological validity of real, templated, and LLM-generated prompts in a small-scale study covering 3 highly contested policy issues and 3 recent geopolitical conflicts. 

Human and LLM annotators rank LLM-generated prompts as no less realistic than real ones and clearly more realistic than templated ones, and find that they carry their intended intent and stance more clearly; the LLMs separate templated prompts from the other two far more sharply than the humans do. In a case study, templated and LLM-generated prompts yield systematically different stance estimates for the same model, most visibly under neutral framings, where templated prompts overstate the model's leaning in the direction encoded by the topic-and-stance text (filler) slotted into their templates.


## Configs

- **prompts** (5992 rows): one row per unique prompt -- arm (method), topic, intent, stance, prompt text, gpt/grok responses, and the 3-judge panel's stance label for the gpt/grok response.
- **realness_ranking** (120 rows): the realness-ranking task -- one row per triple (one real, one templated, one llm-generated prompt each) from 120 such triples, with 3 human raters' and 3 LLM judges' 1-3 "most likely typed by a real person" ranks as separate columns per rater.
- **detection_annotations** (150 rows): topic/intent/stance detection task -- one row per prompt from a 150-prompt subset, with 2 human annotators' and 2 LLM annotators' independently-assigned topic/intent/stance as separate columns per rater.
- **topics_details** (6 rows): topic definitions -- title, description, and the two poles (title + full description + associated loaded terms) each topic's stance is measured against.

## Topics, intents, stances

**Topics and Poles**:

- `climate change` -- Climate-urgency vs. Climate-moderation
- `immigration` -- Immigration-expansive vs. Immigration-restrictive
- `artificial intelligence` -- AI-positivism vs. AI-scepticism
- `israel-palestine` -- Pro-Israel vs. Pro-Palestine
- `russia-ukraine` -- Pro-Ukraine vs. Pro-Russia
- `us/israel-iran` -- Pro-US/Israel vs. Pro-Iran

**Intents**: `information_seeking`, `opinion_sharing`, `writing_assistance`

**Stance**: `neutral`, then the topic's pole A title, then its pole B title (e.g. for climate change: `neutral`, `climate-urgency`, `climate-moderation`) -- see topics_details for every topic's pole names.

## Loading the datasets

```python
from datasets import load_dataset

prompts = load_dataset("kiddothe2b/synthetic_polistance", "prompts")
detection_annotations = load_dataset("kiddothe2b/synthetic_polistance", "detection_annotations")
realness_ranking = load_dataset("kiddothe2b/synthetic_polistance", "realness_ranking")
topics_details = load_dataset("kiddothe2b/synthetic_polistance", "topics_details")
```

## Contact

[author_firstname] [dot] [author_lastname] [at] di [dot] ku [dot] dk

## Citation Information

If you use any of the provided resources, please cite the paper:

```
@misc{chalkidis_2026_synthetic_polistance,
      title={Templated or fully Synthetic? Prompt construction as a confound in measuring LLM political stance beyond writing assistance}, 
      author={Ilias Chalkidis},
      year={2026},
      eprint={2608.11008},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2608.11008}, 
}
```