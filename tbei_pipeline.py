"""
================================================================================
Textual Butterfly Effect Index (TBEI)
Distance-Aware Sensitivity Metric for NLP
================================================================================

Official implementation accompanying:

    Rezaei, A., & Bastanfard, A. (2026).
    Textual Butterfly Effect Index: A Distance-Aware Sensitivity Metric for NLP.

Scope and limitations
---------------------
This is an exploratory research prototype. It does NOT establish:
  - mathematical chaos or a dynamical butterfly effect
  - independent semantic validation
  - generalization across models, domains, or intervention types

The findings are limited to:
  - one NLI model: cross-encoder/nli-MiniLM2-L6-H768
  - one domain: ContractNLI
  - one intervention type: single-token negation insertion

================================================================================
"""

from __future__ import annotations

import json
import math
import hashlib
import time
import re
import zipfile
import random
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from scipy.special import rel_entr
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "project_name": "Textual Butterfly Effect Index",
    "version": "0.1.0",

    "model_id": "cross-encoder/nli-MiniLM2-L6-H768",
    "model_revision": "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
    "max_length": 256,
    "batch_size": 8,

    "far_threshold": 0.30,
    "near_threshold": 0.10,
    "medium_threshold": 0.30,

    "min_span_words": 5,
    "max_span_words": 100,
    "max_context_regex_tokens": 180,

    "seed": 20261031,
    "bootstrap_iterations": 10000,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "torch_threads": 4,

    "modals_priority": ["shall", "must", "will", "may", "is", "are"],
}

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw" / "ContractNLI_official"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
CHECKPOINTS = RESULTS / "checkpoints"

for directory in [DATA_RAW, DATA_PROCESSED, RESULTS, TABLES, FIGURES, CHECKPOINTS]:
    directory.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(CONFIG["torch_threads"])
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

CONTRACTNLI_URL = (
    "https://raw.githubusercontent.com/stanfordnlp/contract-nli/master/"
    "resources/contract-nli.zip"
)
CONTRACTNLI_ZIP = DATA_RAW / "contract-nli-official.zip"
CONTRACTNLI_SHA256 = "0d22e4b1ce9d0f7a0dcab09db59d3e9d6cf93c8f4c7d0a8d6b48c8e8f3c3e0a1"  # update after first download


# ============================================================================
# UTILITIES
# ============================================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(str(type(obj)))


def save_json(data, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)


def load_jsonl_zip(path: Path):
    records = []
    with zipfile.ZipFile(path, "r") as archive:
        members = [
            name for name in archive.namelist()
            if name.endswith(".jsonl") and not name.startswith("__MACOSX/")
        ]
        for member in members:
            with archive.open(member) as stream:
                for line in stream:
                    records.append(json.loads(line.decode("utf-8")))
    return records


def load_official_split(split: str) -> dict:
    with zipfile.ZipFile(CONTRACTNLI_ZIP) as z:
        member = next(
            name for name in z.namelist()
            if name.endswith(f"/{split}.json") or name == f"{split}.json"
        )
        return json.loads(z.read(member).decode("utf-8"))


def normalize_probabilities(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    return p / p.sum(axis=-1, keepdims=True)


def js_normalized(p, q):
    """Normalized Jensen–Shannon divergence in [0, 1] (base-2)."""
    p = normalize_probabilities(p)
    q = normalize_probabilities(q)
    m = 0.5 * (p + q)
    return (
        np.sum(rel_entr(p, m), axis=-1)
        + np.sum(rel_entr(q, m), axis=-1)
    ) / (2 * np.log(2))


# ============================================================================
# STEP 0 — DATA DOWNLOAD
# ============================================================================

def download_contractnli():
    """
    Download the official ContractNLI archive (Koreeda & Manning, 2021).

    Dataset page: https://stanfordnlp.github.io/contract-nli/
    Repository: https://github.com/stanfordnlp/contract-nli
    """
    import requests

    if CONTRACTNLI_ZIP.exists():
        print(f"[OK] Archive already present: {CONTRACTNLI_ZIP}")
        print(f"     SHA256: {sha256_file(CONTRACTNLI_ZIP)}")
        return

    print(f"[INFO] Downloading ContractNLI from {CONTRACTNLI_URL}")
    with requests.get(CONTRACTNLI_URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(CONTRACTNLI_ZIP, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"[OK] Saved: {CONTRACTNLI_ZIP}")
    print(f"     SHA256: {sha256_file(CONTRACTNLI_ZIP)}")


# ============================================================================
# STEP 1 — CONTROLLED SINGLE-TOKEN INTERVENTIONS
# ============================================================================

MODAL_PATTERN = re.compile(
    r"\b(shall|must|may|is|are|will)\b(?!\s+not\b)",
    flags=re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(s: str) -> list[str]:
    return TOKEN_PATTERN.findall(s)


def single_token_insertion(before: str, after: str) -> bool:
    a = tokenize(before)
    b = tokenize(after)
    if len(b) != len(a) + 1:
        return False
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif not skipped:
            skipped = True
            j += 1
        else:
            return False
    return True


def build_interventions(split: str) -> pd.DataFrame:
    """
    Insert a single negation token after the highest-priority modal
    in each eligible evidence span.
    """
    data = load_official_split(split)
    priority = {m: i for i, m in enumerate(CONFIG["modals_priority"])}
    interventions = []

    for doc in data["documents"]:
        text = doc["text"]
        spans = doc["spans"]
        doc_id = doc["id"]

        annotations = doc["annotation_sets"][0]["annotations"]

        evidence_indices = sorted({
            ref
            for ann in annotations.values()
            for ref in ann.get("spans", [])
            if isinstance(ref, int) and 0 <= ref < len(spans)
        })

        for idx in evidence_indices:
            a, b = spans[idx]
            fragment = text[a:b]

            if not (
                CONFIG["min_span_words"] <= len(fragment.split())
                <= CONFIG["max_span_words"]
                and len(fragment) >= 20
            ):
                continue

            matches = list(MODAL_PATTERN.finditer(fragment))
            if not matches:
                continue

            matches.sort(key=lambda m: (priority.get(m.group(1).lower(), 99), m.start()))
            match = matches[0]

            insertion_point = a + match.end()
            modified_text = text[:insertion_point] + " not" + text[insertion_point:]
            modified_fragment = (
                fragment[: match.end()] + " not" + fragment[match.end():]
            )

            if not single_token_insertion(fragment, modified_fragment):
                continue

            interventions.append({
                "intervention_id": len(interventions),
                "split": split,
                "document_id": doc_id,
                "span_index": idx,
                "span_start": a,
                "span_end": b,
                "insertion_char_position": insertion_point,
                "relative_position": insertion_point / len(text),
                "modal": match.group(1).lower(),
                "original_fragment": fragment,
                "modified_fragment": modified_fragment,
                "original_text_sha256": sha256_text(text),
                "modified_text_sha256": sha256_text(modified_text),
            })

    df = pd.DataFrame(interventions)
    print(f"[OK] {split}: {len(df)} interventions from "
          f"{df['document_id'].nunique()} documents")
    return df


# ============================================================================
# STEP 2 — EVIDENCE-GROUNDED NLI INPUT CONSTRUCTION
# ============================================================================

def build_nli_inputs(split: str, interventions: pd.DataFrame) -> pd.DataFrame:
    """
    Build (before_context, after_context, hypothesis) triples for each
    (intervention, observation) pair that fits the model token budget.
    """
    data = load_official_split(split)

    documents = {str(doc["id"]): doc for doc in data["documents"]}
    hypotheses = {
        str(k): v["hypothesis"] for k, v in data["labels"].items()
    }

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_id"],
        revision=CONFIG["model_revision"],
    )

    records = []
    rejection = Counter()

    for edit in interventions.itertuples(index=False):
        doc = documents[str(edit.document_id)]
        text = doc["text"]
        spans = doc["spans"]

        edited_idx = int(edit.span_index)
        a1, b1 = spans[edited_idx]
        original_fragment = text[a1:b1]
        modified_fragment = edit.modified_fragment
        edited_center = (a1 + b1) / 2

        evidence_map = defaultdict(set)
        for annotation_set in doc.get("annotation_sets", []):
            for hypothesis_id, annotation in annotation_set.get("annotations", {}).items():
                for ref in annotation.get("spans", []):
                    if isinstance(ref, int) and 0 <= ref < len(spans):
                        evidence_map[ref].add(str(hypothesis_id))

        for obs_idx, hypothesis_ids in evidence_map.items():
            if obs_idx == edited_idx:
                continue

            a2, b2 = spans[obs_idx]
            observation = text[a2:b2]

            if not (
                CONFIG["min_span_words"] <= len(observation.split())
                <= CONFIG["max_span_words"]
            ):
                rejection["observation_length"] += 1
                continue

            obs_center = (a2 + b2) / 2
            distance = abs(obs_center - edited_center) / max(len(text), 1)

            if distance < CONFIG["near_threshold"]:
                group = "near"
            elif distance < CONFIG["medium_threshold"]:
                group = "medium"
            else:
                group = "far"

            before = original_fragment + "\n\n" + observation
            after = modified_fragment + "\n\n" + observation

            if (
                len(tokenize(before)) > CONFIG["max_context_regex_tokens"]
                or len(tokenize(after)) > CONFIG["max_context_regex_tokens"]
            ):
                rejection["regex_limit"] += 1
                continue

            for hypothesis_id in sorted(hypothesis_ids):
                if hypothesis_id not in hypotheses:
                    continue

                hypothesis = hypotheses[hypothesis_id]

                before_len = len(tokenizer(before, hypothesis, truncation=False)["input_ids"])
                after_len = len(tokenizer(after, hypothesis, truncation=False)["input_ids"])

                if max(before_len, after_len) > CONFIG["max_length"]:
                    rejection["model_limit"] += 1
                    continue

                records.append({
                    "intervention_id": int(edit.intervention_id),
                    "document_id": str(edit.document_id),
                    "edited_span_index": edited_idx,
                    "observation_span_index": obs_idx,
                    "hypothesis_id": hypothesis_id,
                    "hypothesis_text": hypothesis,
                    "before_context": before,
                    "after_context": after,
                    "before_token_length": before_len,
                    "after_token_length": after_len,
                    "relative_distance": float(distance),
                    "distance_group": group,
                    "modal": edit.modal,
                })

    df = pd.DataFrame(records)
    print(f"[OK] {split}: {len(df)} NLI inputs; "
          f"rejections: {dict(rejection)}")
    return df


# ============================================================================
# STEP 3 — NLI INFERENCE
# ============================================================================

def run_inference(inputs: pd.DataFrame, split: str) -> pd.DataFrame:
    """
    Evaluate before/after premises with the NLI model and compute
    pair-level normalized JS divergence. Checkpoints every 200 pairs.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_id"],
        revision=CONFIG["model_revision"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_id"],
        revision=CONFIG["model_revision"],
    )
    model.to(CONFIG["device"])
    model.eval()

    checkpoint_dir = CHECKPOINTS / f"{split}_inference"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    completed = set()
    for path in sorted(checkpoint_dir.glob("ckpt_*.parquet")):
        completed.update(pd.read_parquet(path)["pair_index"].astype(int).tolist())

    pending = [i for i in range(len(inputs)) if i not in completed]
    print(f"[INFO] {split}: {len(pending)} pending pairs")

    buffer = []
    checkpoint_number = len(list(checkpoint_dir.glob("ckpt_*.parquet")))

    def infer_batch(premises, hypotheses):
        encoded = tokenizer(
            premises, hypotheses,
            padding=True, truncation=False,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > CONFIG["max_length"]:
            raise RuntimeError("Input exceeds MAX_LENGTH")
        encoded = {k: v.to(CONFIG["device"]) for k, v in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return normalize_probabilities(probs)

    for start in range(0, len(pending), CONFIG["batch_size"]):
        batch_indices = pending[start:start + CONFIG["batch_size"]]
        batch = inputs.iloc[batch_indices]

        before_probs = infer_batch(
            batch["before_context"].tolist(),
            batch["hypothesis_text"].tolist(),
        )
        after_probs = infer_batch(
            batch["after_context"].tolist(),
            batch["hypothesis_text"].tolist(),
        )

        js_values = js_normalized(before_probs, after_probs)

        for local_i, pair_index in enumerate(batch_indices):
            row = inputs.iloc[pair_index]
            buffer.append({
                "pair_index": int(pair_index),
                "intervention_id": int(row["intervention_id"]),
                "document_id": str(row["document_id"]),
                "edited_span_index": int(row["edited_span_index"]),
                "observation_span_index": int(row["observation_span_index"]),
                "hypothesis_id": str(row["hypothesis_id"]),
                "relative_distance": float(row["relative_distance"]),
                "distance_group": row["distance_group"],
                "modal": row["modal"],
                "js_normalized": float(js_values[local_i]),
                "before_prob_0": float(before_probs[local_i, 0]),
                "before_prob_1": float(before_probs[local_i, 1]),
                "before_prob_2": float(before_probs[local_i, 2]),
                "after_prob_0": float(after_probs[local_i, 0]),
                "after_prob_1": float(after_probs[local_i, 1]),
                "after_prob_2": float(after_probs[local_i, 2]),
                "before_label": int(np.argmax(before_probs[local_i])),
                "after_label": int(np.argmax(after_probs[local_i])),
            })

        if len(buffer) >= 200 or start + CONFIG["batch_size"] >= len(pending):
            checkpoint_number += 1
            path = checkpoint_dir / f"ckpt_{checkpoint_number:05d}.parquet"
            pd.DataFrame(buffer).to_parquet(path, index=False)
            buffer = []
            done = len(completed) + min(start + CONFIG["batch_size"], len(pending))
            print(f"  {split}: {done}/{len(inputs)} pairs done")

    results = pd.concat(
        [pd.read_parquet(p) for p in sorted(checkpoint_dir.glob("ckpt_*.parquet"))],
        ignore_index=True,
    )
    results = (
        results.drop_duplicates("pair_index", keep="first")
        .sort_values("pair_index")
        .reset_index(drop=True)
    )
    assert len(results) == len(inputs), f"Incomplete: {len(results)}/{len(inputs)}"
    return results


# ============================================================================
# STEP 4 — INTERVENTION-LEVEL TBEI SCORES
# ============================================================================

def compute_tbei_scores(pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pair-level JS to intervention-level scores.

        MeanJS(k)     = mean(Δ_j)
        TBEI_raw(k)   = mean(d_j * Δ_j)
        TBEI_norm(k)  = sum(d_j * Δ_j) / sum(d_j)
        NPR(k)        = TBEI_norm(k) / MeanJS(k)
    """
    rows = []
    for intervention_id, group in pairs.groupby("intervention_id"):
        delta = group["js_normalized"].to_numpy(dtype=np.float64)
        distance = group["relative_distance"].to_numpy(dtype=np.float64)

        mean_js = float(delta.mean())
        tbei_raw = float(np.mean(distance * delta))
        d_sum = float(distance.sum())
        tbei_norm = float(np.sum(distance * delta) / d_sum) if d_sum > 0 else np.nan
        npr = float(tbei_norm / mean_js) if mean_js > 1e-12 else np.nan

        far = distance >= CONFIG["far_threshold"]
        flip = group["before_label"].to_numpy() != group["after_label"].to_numpy()

        rows.append({
            "intervention_id": int(intervention_id),
            "document_id": group["document_id"].iloc[0],
            "modal": group["modal"].iloc[0],
            "observation_count": len(group),
            "mean_js": mean_js,
            "tbei_raw": tbei_raw,
            "tbei_norm": tbei_norm,
            "npr": npr,
            "max_js": float(delta.max()),
            "mean_distance": float(distance.mean()),
            "label_flip_rate": float(flip.mean()),
            "any_far_flip": int(np.any(flip[far])) if far.any() else np.nan,
            "far_observations": int(far.sum()),
        })
    return pd.DataFrame(rows)


# ============================================================================
# STEP 5 — STATISTICAL VALIDATION
# ============================================================================

def bootstrap_mean_ci(values, iterations=None, seed=None):
    iterations = iterations or CONFIG["bootstrap_iterations"]
    seed = seed if seed is not None else CONFIG["seed"]
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def document_bootstrap(scores: pd.DataFrame, columns: list[str],
                       iterations=None, seed=None) -> dict:
    iterations = iterations or CONFIG["bootstrap_iterations"]
    seed = seed if seed is not None else CONFIG["seed"]
    rng = np.random.default_rng(seed)

    doc_means = scores.groupby("document_id")[columns].mean().dropna()
    values = doc_means.to_numpy(dtype=np.float64)
    n = len(values)

    results = {}
    for j, column in enumerate(columns):
        observed = float(values[:, j].mean())
        draws = np.empty(iterations)
        for b in range(iterations):
            idx = rng.integers(0, n, size=n)
            draws[b] = values[idx, j].mean()
        results[column] = {
            "document_mean": observed,
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "documents": n,
        }
    return results


def safe_spearman(x, y):
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return {"rho": None, "p_value": None}
    r = spearmanr(a[mask], b[mask])
    return {"rho": float(r.statistic), "p_value": float(r.pvalue)}


# ============================================================================
# STEP 6 — INCREMENTAL PREDICTIVE UTILITY (LOGISTIC REGRESSION)
# ============================================================================

FEATURE_SETS = {
    "Baseline": ["observation_count", "modal"],
    "Baseline + Mean JS": ["observation_count", "modal", "mean_js"],
    "Baseline + TBEI": ["observation_count", "modal", "tbei_raw"],
    "Baseline + JS + Distance": [
        "observation_count", "modal", "mean_js", "mean_distance",
    ],
    "Baseline + JS + Distance + TBEI": [
        "observation_count", "modal", "mean_js", "mean_distance", "tbei_raw",
    ],
}


def build_logistic_pipeline(columns: list[str]) -> Pipeline:
    numeric = [c for c in columns if c != "modal"]
    categorical = [c for c in columns if c == "modal"]

    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical,
        ))

    return Pipeline([
        ("preprocessor", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
            max_iter=3000,
            random_state=CONFIG["seed"],
        )),
    ])


def train_and_evaluate_logistic(dev: pd.DataFrame, test: pd.DataFrame):
    y_dev = dev["any_far_flip"].astype(int).to_numpy()
    y_test = test["any_far_flip"].astype(int).to_numpy()

    results = []
    probabilities = {}

    for name, columns in FEATURE_SETS.items():
        pipeline = build_logistic_pipeline(columns)
        pipeline.fit(dev[columns], y_dev)
        prob = np.clip(pipeline.predict_proba(test[columns])[:, 1], 1e-9, 1 - 1e-9)
        probabilities[name] = prob

        results.append({
            "model": name,
            "features": ", ".join(columns),
            "auroc": float(roc_auc_score(y_test, prob)),
            "ap": float(average_precision_score(y_test, prob)),
            "brier": float(brier_score_loss(y_test, prob)),
            "logloss": float(log_loss(y_test, prob)),
        })

    return pd.DataFrame(results), probabilities


def cluster_bootstrap_comparison(test_df: pd.DataFrame,
                                 probabilities: dict,
                                 reference: str, improved: str,
                                 iterations=None) -> dict:
    iterations = iterations or CONFIG["bootstrap_iterations"]
    rng = np.random.default_rng(CONFIG["seed"])

    y = test_df["any_far_flip"].astype(int).to_numpy()
    docs = test_df["document_id"].astype(str).to_numpy()
    unique_docs = np.unique(docs)
    doc_indices = {d: np.flatnonzero(docs == d) for d in unique_docs}

    deltas = {"auroc": [], "ap": [], "brier": [], "logloss": []}

    for _ in range(iterations):
        sampled = rng.choice(unique_docs, size=len(unique_docs), replace=True)
        idx = np.concatenate([doc_indices[d] for d in sampled])
        yy = y[idx]
        if np.unique(yy).size < 2:
            continue

        p_ref = probabilities[reference][idx]
        p_new = probabilities[improved][idx]

        deltas["auroc"].append(
            roc_auc_score(yy, p_new) - roc_auc_score(yy, p_ref)
        )
        deltas["ap"].append(
            average_precision_score(yy, p_new)
            - average_precision_score(yy, p_ref)
        )
        deltas["brier"].append(
            brier_score_loss(yy, p_new) - brier_score_loss(yy, p_ref)
        )
        deltas["logloss"].append(
            log_loss(yy, p_new) - log_loss(yy, p_ref)
        )

    observed = {
        "auroc": float(
            roc_auc_score(y, probabilities[improved])
            - roc_auc_score(y, probabilities[reference])
        ),
        "ap": float(
            average_precision_score(y, probabilities[improved])
            - average_precision_score(y, probabilities[reference])
        ),
        "brier": float(
            brier_score_loss(y, probabilities[improved])
            - brier_score_loss(y, probabilities[reference])
        ),
        "logloss": float(
            log_loss(y, probabilities[improved])
            - log_loss(y, probabilities[reference])
        ),
    }

    summary = {}
    for metric, values in deltas.items():
        arr = np.asarray(values, dtype=float)
        summary[metric] = {
            "observed_delta": observed[metric],
            "ci95_low": float(np.quantile(arr, 0.025)),
            "ci95_high": float(np.quantile(arr, 0.975)),
            "iterations": len(arr),
        }
    return summary


# ============================================================================
# STEP 7 — CONTROLS (PERMUTATION, WHITESPACE, ORDER)
# ============================================================================

def distance_permutation_control(scores: pd.DataFrame,
                                 pairs: pd.DataFrame,
                                 permutations=None) -> dict:
    """
    Shuffle the distance vector within each intervention and recompute TBEI.
    This is a structural diagnostic, NOT independent semantic validation.
    """
    permutations = permutations or 2000
    rng = np.random.default_rng(CONFIG["seed"])

    eval_df = scores.dropna(subset=["any_far_flip"]).copy()
    y = eval_df["any_far_flip"].astype(int).to_numpy()
    mean_js = eval_df["mean_js"].to_numpy()
    real_tbei = eval_df["tbei_raw"].to_numpy()

    groups = {
        iid: pairs[pairs["intervention_id"] == iid]
        for iid in eval_df["intervention_id"]
    }

    real_metrics = {
        "mean_js_auroc": float(roc_auc_score(y, mean_js)),
        "mean_js_ap": float(average_precision_score(y, mean_js)),
        "tbei_auroc": float(roc_auc_score(y, real_tbei)),
        "tbei_ap": float(average_precision_score(y, real_tbei)),
    }

    perm_auc = np.empty(permutations)
    perm_ap = np.empty(permutations)

    for b in range(permutations):
        shuffled = np.empty(len(eval_df))
        for i, iid in enumerate(eval_df["intervention_id"]):
            g = groups[iid]
            d = g["relative_distance"].to_numpy()
            j = g["js_normalized"].to_numpy()
            shuffled[i] = np.mean(rng.permutation(d) * j)
        perm_auc[b] = roc_auc_score(y, shuffled)
        perm_ap[b] = average_precision_score(y, shuffled)

    return {
        "real": real_metrics,
        "permuted_auroc_mean": float(perm_auc.mean()),
        "permuted_auroc_ci95": np.quantile(perm_auc, [0.025, 0.975]).tolist(),
        "permuted_ap_mean": float(perm_ap.mean()),
        "permuted_ap_ci95": np.quantile(perm_ap, [0.025, 0.975]).tolist(),
        "upper_tail_p_auroc": float(
            (1 + np.sum(perm_auc >= real_metrics["tbei_auroc"]))
            / (permutations + 1)
        ),
        "upper_tail_p_ap": float(
            (1 + np.sum(perm_ap >= real_metrics["tbei_ap"]))
            / (permutations + 1)
        ),
        "warning": (
            "The far-flip target is defined using actual distances. "
            "This control is a structural diagnostic, not semantic validation."
        ),
    }


def whitespace_control(inputs: pd.DataFrame, max_pairs: int = 1200) -> dict:
    """
    Replace one ordinary space with two spaces (tokenization-preserving)
    and measure response change.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_id"], revision=CONFIG["model_revision"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_id"], revision=CONFIG["model_revision"],
    )
    model.to(CONFIG["device"])
    model.eval()

    rng = np.random.default_rng(CONFIG["seed"] + 31)

    def split_context(before, after):
        if not isinstance(before, str) or not isinstance(after, str):
            return None
        if "\n\n" not in before or "\n\n" not in after:
            return None
        b1, b2 = before.split("\n\n", 1)
        a1, a2 = after.split("\n\n", 1)
        if b2 != a2 or b1 == a1:
            return None
        return b1, a1, b2

    inputs = inputs.copy()
    inputs["components"] = [
        split_context(b, a)
        for b, a in zip(inputs["before_context"], inputs["after_context"])
    ]
    valid = inputs[inputs["components"].notna()].reset_index(drop=True)

    if len(valid) == 0:
        return {"error": "No valid two-fragment inputs."}

    sample = valid.sample(
        n=min(max_pairs, len(valid)),
        random_state=CONFIG["seed"],
    ).reset_index(drop=True)

    def ws_edit(text):
        for i in range(1, len(text) - 1):
            if text[i] == " " and text[i - 1] != " " and text[i + 1] != " ":
                return text[:i] + "  " + text[i + 1:]
        return None

    requests, keys = [], []
    for i, row in sample.iterrows():
        orig, edited, obs = row["components"]
        neg = ws_edit(orig)
        if neg is None:
            continue
        requests.append((orig + "\n\n" + obs, row["hypothesis_text"]))
        keys.append((i, "original"))
        requests.append((neg + "\n\n" + obs, row["hypothesis_text"]))
        keys.append((i, "whitespace"))

    if not requests:
        return {"error": "No whitespace variants could be generated."}

    def infer(premises, hyps):
        out = []
        for start in range(0, len(premises), CONFIG["batch_size"]):
            enc = tokenizer(
                premises[start:start + CONFIG["batch_size"]],
                hyps[start:start + CONFIG["batch_size"]],
                padding=True, truncation=True,
                max_length=CONFIG["max_length"],
                return_tensors="pt",
            )
            enc = {k: v.to(CONFIG["device"]) for k, v in enc.items()}
            with torch.inference_mode():
                probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            out.append(normalize_probabilities(probs))
        return np.vstack(out)

    probs = infer([r[0] for r in requests], [r[1] for r in requests])
    lookup = {k: probs[i] for i, k in enumerate(keys)}

    js_values, flips = [], []
    for i in sample.index:
        p_orig = lookup.get((i, "original"))
        p_ws = lookup.get((i, "whitespace"))
        if p_orig is None or p_ws is None:
            continue
        js_values.append(float(js_normalized(p_orig, p_ws)))
        flips.append(int(np.argmax(p_orig) != np.argmax(p_ws)))

    if not js_values:
        return {"error": "No whitespace-control pairs computed."}

    return {
        "pairs": len(js_values),
        "mean_js": float(np.mean(js_values)),
        "median_js": float(np.median(js_values)),
        "p95_js": float(np.quantile(js_values, 0.95)),
        "label_flip_rate": float(np.mean(flips)),
    }


def order_control(inputs: pd.DataFrame, max_pairs: int = 1200) -> dict:
    """
    Reverse the two-fragment premise order and recompute JS.
    Reversal is not equivalent to restoring original document order.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_id"], revision=CONFIG["model_revision"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_id"], revision=CONFIG["model_revision"],
    )
    model.to(CONFIG["device"])
    model.eval()

    def split_context(before, after):
        if not isinstance(before, str) or not isinstance(after, str):
            return None
        if "\n\n" not in before or "\n\n" not in after:
            return None
        b1, b2 = before.split("\n\n", 1)
        a1, a2 = after.split("\n\n", 1)
        if b2 != a2 or b1 == a1:
            return None
        return b1, a1, b2

    inputs = inputs.copy()
    inputs["components"] = [
        split_context(b, a)
        for b, a in zip(inputs["before_context"], inputs["after_context"])
    ]
    valid = inputs[inputs["components"].notna()].reset_index(drop=True)

    if len(valid) == 0:
        return {"error": "No valid two-fragment inputs."}

    sample = valid.sample(
        n=min(max_pairs, len(valid)),
        random_state=CONFIG["seed"],
    ).reset_index(drop=True)

    premises, keys = [], []
    for i, row in sample.iterrows():
        orig, edited, obs = row["components"]
        premises.extend([
            orig + "\n\n" + obs,
            edited + "\n\n" + obs,
            obs + "\n\n" + orig,
            obs + "\n\n" + edited,
        ])
        keys.extend([
            (i, "forward_before"),
            (i, "forward_after"),
            (i, "reverse_before"),
            (i, "reverse_after"),
        ])

    def infer(premises, hyps):
        out = []
        for start in range(0, len(premises), CONFIG["batch_size"]):
            enc = tokenizer(
                premises[start:start + CONFIG["batch_size"]],
                hyps[start:start + CONFIG["batch_size"]],
                padding=True, truncation=True,
                max_length=CONFIG["max_length"],
                return_tensors="pt",
            )
            enc = {k: v.to(CONFIG["device"]) for k, v in enc.items()}
            with torch.inference_mode():
                probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            out.append(normalize_probabilities(probs))
        return np.vstack(out)

    probs = infer(premises, [sample.iloc[k[0]]["hypothesis_text"] for k in keys])
    lookup = {k: probs[i] for i, k in enumerate(keys)}

    forward_js, reverse_js, disagreements = [], [], []
    for i in sample.index:
        fb = lookup.get((i, "forward_before"))
        fa = lookup.get((i, "forward_after"))
        rb = lookup.get((i, "reverse_before"))
        ra = lookup.get((i, "reverse_after"))
        if any(x is None for x in [fb, fa, rb, ra]):
            continue
        f_js = float(js_normalized(fb, fa))
        r_js = float(js_normalized(rb, ra))
        forward_js.append(f_js)
        reverse_js.append(r_js)
        f_flip = int(np.argmax(fb) != np.argmax(fa))
        r_flip = int(np.argmax(rb) != np.argmax(ra))
        disagreements.append(int(f_flip != r_flip))

    if not forward_js:
        return {"error": "No order-control pairs computed."}

    return {
        "pairs": len(forward_js),
        "forward_mean_js": float(np.mean(forward_js)),
        "reverse_mean_js": float(np.mean(reverse_js)),
        "mean_js_difference": float(np.mean(np.array(reverse_js) - np.array(forward_js))),
        "flip_disagreement_rate": float(np.mean(disagreements)),
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("=" * 78)
    print(f"{CONFIG['project_name']} — v{CONFIG['version']}")
    print("=" * 78)

    # ------------------------------------------------------------------
    # 0. Download data
    # ------------------------------------------------------------------
    print("\n[0] Data")
    download_contractnli()

    # ------------------------------------------------------------------
    # 1. Build interventions
    # ------------------------------------------------------------------
    print("\n[1] Controlled interventions")
    interventions = {}
    for split in ["dev", "test"]:
        interventions[split] = build_interventions(split)
        interventions[split].to_parquet(
            DATA_PROCESSED / f"{split}_interventions.parquet", index=False
        )

    # ------------------------------------------------------------------
    # 2. Build NLI inputs
    # ------------------------------------------------------------------
    print("\n[2] NLI inputs")
    nli_inputs = {}
    for split in ["dev", "test"]:
        nli_inputs[split] = build_nli_inputs(split, interventions[split])
        nli_inputs[split].to_parquet(
            DATA_PROCESSED / f"{split}_nli_inputs.parquet", index=False
        )

    # ------------------------------------------------------------------
    # 3. Run NLI inference
    # ------------------------------------------------------------------
    print("\n[3] Inference")
    pair_responses = {}
    for split in ["dev", "test"]:
        pair_responses[split] = run_inference(nli_inputs[split], split)
        pair_responses[split].to_parquet(
            DATA_PROCESSED / f"{split}_pair_responses.parquet", index=False
        )

    # ------------------------------------------------------------------
    # 4. Compute TBEI scores
    # ------------------------------------------------------------------
    print("\n[4] TBEI scores")
    scores = {}
    for split in ["dev", "test"]:
        scores[split] = compute_tbei_scores(pair_responses[split])
        scores[split].to_parquet(
            DATA_PROCESSED / f"{split}_tbei_scores.parquet", index=False
        )

    # ------------------------------------------------------------------
    # 5. Direct predictive utility
    # ------------------------------------------------------------------
    print("\n[5] Direct predictive utility")
    utility_rows = []
    for split in ["dev", "test"]:
        df = scores[split].dropna(subset=["any_far_flip"])
        y = df["any_far_flip"].astype(int).to_numpy()
        for name, column in [("MeanJS", "mean_js"), ("TBEI_raw", "tbei_raw")]:
            utility_rows.append({
                "split": split,
                "metric": name,
                "n_interventions": len(df),
                "n_documents": df["document_id"].nunique(),
                "prevalence": float(y.mean()),
                "auroc": float(roc_auc_score(y, df[column])),
                "average_precision": float(average_precision_score(y, df[column])),
            })
    direct_utility = pd.DataFrame(utility_rows)
    direct_utility.to_csv(TABLES / "direct_utility.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Logistic regression (incremental utility)
    # ------------------------------------------------------------------
    print("\n[6] Logistic regression")
    dev_df = scores["dev"].dropna(subset=["any_far_flip"])
    test_df = scores["test"].dropna(subset=["any_far_flip"])

    logistic_results, probabilities = train_and_evaluate_logistic(dev_df, test_df)
    logistic_results.to_csv(TABLES / "logistic_results.csv", index=False)

    comparison = cluster_bootstrap_comparison(
        test_df, probabilities,
        reference="Baseline + JS + Distance",
        improved="Baseline + JS + Distance + TBEI",
    )
    save_json(comparison, TABLES / "incremental_utility_ci.json")

    # ------------------------------------------------------------------
    # 7. Controls
    # ------------------------------------------------------------------
    print("\n[7] Controls")
    permutation = distance_permutation_control(
        scores["test"], pair_responses["test"]
    )
    save_json(permutation, TABLES / "permutation_control.json")

    whitespace = whitespace_control(nli_inputs["test"])
    save_json(whitespace, TABLES / "whitespace_control.json")

    order = order_control(nli_inputs["test"])
    save_json(order, TABLES / "order_control.json")

    # ------------------------------------------------------------------
    # 8. Figures
    # ------------------------------------------------------------------
    print("\n[8] Figures")
    make_figures(scores["test"], pair_responses["test"])

    print("\n" + "=" * 78)
    print("DONE — results in:", TABLES)
    print("=" * 78)


def make_figures(scores: pd.DataFrame, pairs: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Figure: spatial sensitivity signature
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    groups = ["near", "medium", "far"]
    means = [pairs[pairs["distance_group"] == g]["js_normalized"].mean() for g in groups]
    bars = ax.bar(groups, means, color=["#f5931e", "#94ae54", "#5e8333"])
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0005, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean normalized JS divergence")
    ax.set_xlabel("Relative distance group")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_spatial_sensitivity.png", dpi=300)
    fig.savefig(FIGURES / "figure_spatial_sensitivity.pdf")
    plt.close(fig)

    # Figure: distance vs response scatter
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hexbin(
        pairs["relative_distance"], pairs["js_normalized"],
        gridsize=45, mincnt=1, bins="log", cmap="viridis",
    )
    ax.axvline(CONFIG["far_threshold"], linestyle="--", color="#ec3315")
    ax.set_xlabel("Relative document distance")
    ax.set_ylabel("Normalized JS divergence")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure_distance_vs_response.png", dpi=300)
    fig.savefig(FIGURES / "figure_distance_vs_response.pdf")
    plt.close(fig)

    # Figure: ROC/PR curves
    from sklearn.metrics import roc_curve, precision_recall_curve
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    y = scores["any_far_flip"].astype(int).to_numpy()
    for column, label, color in [
        ("mean_js", "Mean JS", "#f5931e"),
        ("tbei_raw", "TBEI (raw)", "#5e8333"),
    ]:
        prob = scores[column].to_numpy()
        fpr, tpr, _ = roc_curve(y, prob)
        prec, rec, _ = precision_recall_curve(y, prob)
        auc = roc_auc_score(y, prob)
        ap = average_precision_score(y, prob)
        axes[0].plot(fpr, tpr, color=color, label=f"{label} ({auc:.3f})")
        axes[1].plot(rec, prec, color=color, label=f"{label} ({ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "--", color="#2b3118", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("(a) ROC")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.15)

    axes[1].axhline(y.mean(), linestyle="--", color="#2b3118", alpha=0.4)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("(b) Precision–Recall")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(FIGURES / "figure_roc_pr.png", dpi=300)
    fig.savefig(FIGURES / "figure_roc_pr.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
