# Empirical Evaluation of Structured Logical Reasoning in Open-Weight Language Models: Effects of Model Size, Task Complexity, and Language Adaptation

Ari Saptawijaya, Vander Gerald Sukandi, Mikhael Deo Barli

This repository contains the codes and data of an independent empirical validation of a decompose-search-resolve framework, Aristotle, under previously unexplored settings involving open-weight large and small language models. 

The codes in this repository are mostly based on the implementation of the referred Aristotle framework: 

Xu, J., Fei, H., Luo, M., Liu, Q., Pan, L., Wang, W. Y., Nakov, P., Lee, M.-L., and Hsu, W. (2025). Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T., editors, *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3052–3075, Vienna, Austria. Association for Computational Linguistics.

with some adaptation for the evaluation purpose in the present work.

**It consists of two evaluation tasks below.** 

## 1. Evaluation and validation on open-weight LLMs
The evaluation is on the two datasets (ProntoQA and ProofWriter) with performance compared against naive and few-shot prompting baselines. 

> **Folder:** openweight-eval

This experiment is built with minimal third-party dependencies.
### External Libraries

Install the required external libraries via pip to support parallel processing and data integrity:

- *tqdm*: Provides a progress bar indicator during large-scale data iteration.
- *filelock*: A critical library for atomically locking JSON file read/write operations (thread-safe), preventing data corruption caused by race conditions when scripts are executed in a multithreaded context.

bash
pip install tqdm filelock


### Local Dependencies

- *utils.py*: The inference scripts depend heavily on this local module, which contains the OpenAIModel class responsible for managing the entire API call pipeline to the LLM. Ensure that utils.py is located in the root directory of your repository.

### Standard Python Libraries

This experiment makes use of Python's standard library (recommended: *Python 3.8+*), including json for data parsing, re for regex manipulation and reasoning-tag stripping, and concurrent.futures along with threading for parallel batching acceleration.

### Usage Instructions

All commands below must be executed from the root directory of your repository. In the instructions, DeepInfra is used as an inference-as-a-service provider.


#### Method 1: True Naive Prompting Baseline

This method evaluates the model's raw logical reasoning capability (baseline) without requiring step-by-step explanation.

Step 1 — Run Inference:

bash
python true_naive_pt.py \
    --dataset_name "ProntoQA" \
    --model_name "Qwen/Qwen3-14B" \
    --base_url "https://api.deepinfra.com/v1/openai" \
    --api_key "YOUR_DEEPINFRA_API_KEY" \
    --batch_num 4


> You may substitute --model_name with Qwen/Qwen3-32B or deepseek-ai/DeepSeek-R1-0528.

Step 2 — Run Evaluation:

bash
python evaluate_true_naive.py \
    --dataset_name "ProntoQA" \
    --model_name "qwen3-14b"


> For all evaluation scripts in the directory, use the shortened name variant: qwen3-14b, qwen3-32b, or deepseek.

---

#### Method 2: Few-shot Prompting Baseline

A second comparative method using a simple In-Context Learning (ICL) approach to guide the model's reasoning process with a few structured examples.

Step 1 — Run Inference:

bash
python fewshot_prompting.py \
    --dataset_name "ProntoQA" \
    --model_name "Qwen/Qwen3-14B" \
    --base_url "https://api.deepinfra.com/v1/openai" \
    --api_key "YOUR_DEEPINFRA_API_KEY" \
    --batch_num 4


Step 2 — Run Evaluation:

bash
python evaluate_fewshot_pt.py \
    --dataset_name "ProntoQA" \
    --model_name "qwen3-14b"


---

#### Method 3: Decompose-Search-Resolve (Aristotle) Framework

The main 'complex' framework mentioned in this paper, executing logic end-to-end through a three-stage architecture: Decompose, Search, and Resolve.

---

Step 1 — Translation and Decomposition

Translates raw premises and decomposes complex formal logical structures.

bash
python translate_decompose.py \
    --dataset_name "ProntoQA" \
    --model_name "Qwen/Qwen3-14B" \
    --base_url "https://api.deepinfra.com/v1/openai" \
    --api_key "YOUR_DEEPINFRA_API_KEY" \
    --split dev \
    --max_new_tokens 6144 \
    --batch_num 4


---

Step 2 — Initialization (Negation Setup)

Prepares the search structure by negating the initial conjecture. Use the shortened name variant: qwen3-14b, qwen3-32b, or deepseek. This evaluation step runs entirely locally and does not require a --base_url parameter.

bash
python negate.py \
    --dataset_name "ProntoQA" \
    --model "qwen3-14b"


---

Step 3 — Search and Resolve

The reasoning graph traversal stage. This step *must be run twice* to separately evaluate the positive resolution path (False) and the negative resolution path (True) using the large language model's inference base.

Run 1 — Non-negation path:

bash
python search_resolve.py \
    --dataset_name "ProntoQA" \
    --model_name "Qwen/Qwen3-14B" \
    --base_url "https://api.deepinfra.com/v1/openai" \
    --api_key "YOUR_DEEPINFRA_API_KEY" \
    --split dev \
    --negation False \
    --max_new_tokens 4096 \
    --batch_num 4


Run 2 — Negation path:

bash
python search_resolve.py \
    --dataset_name "ProntoQA" \
    --model_name "Qwen/Qwen3-14B" \
    --base_url "https://api.deepinfra.com/v1/openai" \
    --api_key "YOUR_DEEPINFRA_API_KEY" \
    --split dev \
    --negation True \
    --max_new_tokens 4096 \
    --batch_num 4


---

Step 4 — Final Conclusion and Evaluation

Aggregates the final decision matrices from Run 1 and Run 2 to derive the ultimate accuracy conclusion based on the logic formula as mentioned in the formula. This evaluation step runs entirely locally and does not require a --base_url parameter. Use the shortened name variant: qwen3-14b, qwen3-32b, or deepseek.

bash
python evaluate.py \
    --dataset_name "ProntoQA" \
    --model "qwen3-14b"

## 2. Cross-lingual evaluation on open-weight small language models
The evaluation is on a translation of ProntoQA dataset into Bahasa Indonesia to examine reasoning behavior in a small-scale, non-English setting. 

> **Folder:** crosslingual-eval

The usage of the codes (including installing the requirements/dependencies) for this study is already set up as Jupyter notebook files in the subfloder *global_runner*. Therein, one runner pipeline is prepared for each model:
- runner_pipeline_qwen.ipynb, for Qwen2.5-7B
- runner_pipeline_sealion.ipynb, for SEA-LION-v3-8B
- runner_pipeline_sahabatai.ipynb, for SahabatAI-v1-8B


