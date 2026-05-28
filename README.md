# Empirical Evaluation of Structured Logical Reasoning in Open-Weight Language Models: Effects of Model Size, Task Complexity, and Language Adaptation

Authors: Ari Saptawijaya, Vander Gerald Sukandi, Mikhael Deo Barli

This repository contains the codes and data of an independent empirical validation of a decompose-search-resolve framework, Aristotle, under previously unexplored settings involving open-weight large and small language models. 

The codes in this repository are mostly based on the implementation of the referred Aristotle framework with some adaptation for the evaluation purpose in the present work.

Xu, J., Fei, H., Luo, M., Liu, Q., Pan, L., Wang, W. Y., Nakov, P., Lee, M.-L., and Hsu, W. (2025). Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T., editors, *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3052–3075, Vienna, Austria. Association for Computational Linguistics.

**It consists of two evaluation tasks:** 

### 1. Evaluation and validation on open-weight LLMs
The evaluation is on the ProntoQA and ProofWriter datasets with performance compared against naive and few-shot prompting baselines. 

**Folder:** openweight-eval

### 2. Cross-lingual evaluation on open-weight small language models
The evaluation is on a translation of ProntoQA into Bahasa Indonesia to examine reasoning behavior in a small-scale, non-English setting. 

**Folder:** crosslingual-eval
