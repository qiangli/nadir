#!/usr/bin/env python3
"""
One-shot exporter that produces every asset the Go ONNX classifier
needs:

    assets/model.onnx           # all-MiniLM-L6-v2 forward pass
    assets/vocab.txt            # BERT WordPiece vocab (pure-Go tokenizer reads this)
    assets/tokenizer_config.json
    assets/simple_centroid.bin  # 384 float32 LE
    assets/complex_centroid.bin # 384 float32 LE
    testdata/classifier_golden.json   # 50-row Python-derived parity fixtures

Run from the repo root:

    python -m venv .venv && source .venv/bin/activate
    pip install -r tools/export-onnx/requirements.txt
    python tools/export-onnx/export.py

The Go ONNX classifier verifies its numerics against
testdata/classifier_golden.json; if you change this script (e.g.
substitute a different embedding model), regenerate the goldens.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 128
EMBED_DIM = 384

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "assets"
TESTDATA = REPO_ROOT / "testdata"
PROTOTYPES_PATH = REPO_ROOT / "priorart" / "NadirClaw" / "nadirclaw" / "prototypes.py"


def load_prototypes() -> tuple[list[str], list[str]]:
    """Load SIMPLE_PROTOTYPES + COMPLEX_PROTOTYPES from the upstream file."""
    spec = importlib.util.spec_from_file_location("prototypes", PROTOTYPES_PATH)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"could not load {PROTOTYPES_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.SIMPLE_PROTOTYPES), list(mod.COMPLEX_PROTOTYPES)


def export_onnx(out_path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval()
    dummy = tokenizer("hello world", padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]),
        str(out_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "token_type_ids": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
        opset_version=14,
    )
    # Vocab + config (the pure-Go tokenizer reads vocab.txt, not
    # tokenizer.json). transformers 5.x writes the vocab to
    # tokenizer.model; the *content* is the same WordPiece text format
    # so we rename it to vocab.txt for clarity.
    saved = tokenizer.save_vocabulary(str(ASSETS))
    if saved:
        for path in saved:
            p = Path(path)
            if p.name != "vocab.txt":
                p.rename(ASSETS / "vocab.txt")
    cfg = {
        "max_len": MAX_LEN,
        "do_lower_case": getattr(tokenizer, "do_lower_case", True),
        "cls_token": tokenizer.cls_token,
        "sep_token": tokenizer.sep_token,
        "pad_token": tokenizer.pad_token,
        "unk_token": tokenizer.unk_token,
    }
    (ASSETS / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2))


def embed_one(model, tokenizer, text: str) -> np.ndarray:
    """Reproduce the production pipeline: tokenize → forward → masked
    mean-pool → L2-normalise."""
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs).last_hidden_state  # [1, L, D]
    mask = inputs["attention_mask"].unsqueeze(-1).float()  # [1, L, 1]
    summed = (out * mask).sum(dim=1)  # [1, D]
    counts = mask.sum(dim=1).clamp(min=1e-9)  # [1, 1]
    pooled = summed / counts  # [1, D]
    pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return pooled.squeeze(0).cpu().numpy().astype(np.float32)


def write_centroid(path: Path, vec: np.ndarray) -> None:
    """Raw little-endian float32 — matches the //go:embed decoder."""
    assert vec.shape == (EMBED_DIM,) and vec.dtype == np.float32, vec.shape
    with open(path, "wb") as f:
        f.write(struct.pack(f"<{EMBED_DIM}f", *vec.tolist()))


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    TESTDATA.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {MODEL_ID} → {ASSETS / 'model.onnx'}")
    export_onnx(ASSETS / "model.onnx")

    print("Loading prototype prompts")
    simple, complex_ = load_prototypes()
    print(f"  {len(simple)} simple, {len(complex_)} complex")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval()

    def centroid(prompts: list[str]) -> np.ndarray:
        vecs = np.stack([embed_one(model, tokenizer, p) for p in prompts], axis=0)
        c = vecs.mean(axis=0)
        return (c / max(float(np.linalg.norm(c)), 1e-9)).astype(np.float32)

    print("Computing centroids")
    s_cen = centroid(simple)
    c_cen = centroid(complex_)
    write_centroid(ASSETS / "simple_centroid.bin", s_cen)
    write_centroid(ASSETS / "complex_centroid.bin", c_cen)
    print(f"  wrote {ASSETS / 'simple_centroid.bin'}")
    print(f"  wrote {ASSETS / 'complex_centroid.bin'}")

    # 50-row golden fixtures: 25 from each prototype list.
    golden = []
    for prompts in (simple[:25], complex_[:25]):
        for p in prompts:
            emb = embed_one(model, tokenizer, p)
            sim_s = float(np.dot(emb, s_cen))
            sim_c = float(np.dot(emb, c_cen))
            confidence = abs(sim_c - sim_s)
            scaled = min(confidence * 5.0, 0.5)
            score = 0.5 + scaled if sim_c > sim_s else 0.5 - scaled
            if score <= 0.35:
                tier = "simple"
            elif score >= 0.65:
                tier = "complex"
            else:
                tier = "simple"  # no mid by default
            golden.append({
                "prompt": p,
                "score": round(score, 6),
                "confidence": round(confidence, 6),
                "tier": tier,
                "sim_simple": round(sim_s, 6),
                "sim_complex": round(sim_c, 6),
            })
    (TESTDATA / "classifier_golden.json").write_text(json.dumps(golden, indent=2))
    print(f"  wrote {TESTDATA / 'classifier_golden.json'}  ({len(golden)} rows)")

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
