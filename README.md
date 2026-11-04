# Textual-Butterfly-Effect-Index
<div align="center">

# 🦋 Textual Butterfly Effect Index (TBEI)

### A Distance-Aware Sensitivity Metric for NLP

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-4.35%2B-FFD21E?style=for-the-badge)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-F59E0B?style=for-the-badge)]()

**Official implementation accompanying:**

> **Textual Butterfly Effect Index: A Distance-Aware Sensitivity Metric for NLP**
> Alireza Rezaei, Azam Bastanfard — *2026*

[Overview](#-overview) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Method](#-method) • [Results](#-results) • [Citation](#-citation)

</div>

---

## ⚠️ Scope and Limitations

<div align="center">

| ✅ What TBEI **is** | ❌ What TBEI **is not** |
|:---|:---|
| A distance-aware aggregation of model-output divergence | A demonstration of mathematical chaos |
| A sensitivity index for controlled local edits | Independent validation of semantic change |
| An exploratory measure of model behavior | A general-purpose robustness benchmark |
| A reproducible research prototype | A claim of state-of-the-art performance |

</div>

> **Important:** The findings are limited to **one NLI model** (`cross-encoder/nli-MiniLM2-L6-H768`), **one domain** (ContractNLI), and **one intervention type** (single-token negation insertion). TBEI and the far-flip target share model responses and distance information; results do not independently validate semantic propagation.

---

## 📖 Overview

**TBEI** measures how a language model's output distribution changes at **unchanged textual observations** after a **local edit**, weighted by the **document distance** between the edit and each observation.

<div align="center">
