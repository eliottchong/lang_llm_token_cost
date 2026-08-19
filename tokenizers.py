#!/usr/bin/env python3
"""Measure a matched corpus with OpenAI o200k, Qwen 2.5, and Mistral 7B.

Install: python -m pip install tiktoken transformers huggingface_hub
Run:     python tokenizers.py

All encodes disable special tokens; no chat template is ever called. Input
strings come from corpus.csv and are normalised to Unicode NFC before
character, byte, and token measurements are calculated.
"""

from __future__ import annotations

# This file is deliberately named ``tokenizers.py``. Remove its directory from
# import lookup before loading Transformers, whose dependency is also named
# ``tokenizers``.
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:] = [path for path in sys.path if Path(path or ".").resolve() != _HERE]

import csv
import hashlib
import json
import unicodedata
from datetime import UTC, datetime
from importlib.metadata import version

import tiktoken
from huggingface_hub import HfApi
from transformers import AutoTokenizer


OUTPUT_JSON = _HERE / "tokenizer_measurements.json"
OUTPUT_CSV = _HERE / "tokenizer_measurements.csv"
CORPUS_CSV = _HERE / "corpus.csv"
REQUIRED_CORPUS_COLUMNS = {
    "item_id", "category", "language", "condition", "text", "adequacy_score"
}

HF_MODELS = {
    "qwen-2.5": {"identifier": "Qwen/Qwen2.5-7B-Instruct", "revision": "main"},
    "mistral-7b": {
        "identifier": "mistralai/Mistral-7B-v0.1",
        "revision": "main",
    },
}


def package_versions() -> dict[str, str]:
    return {
        "tiktoken": version("tiktoken"),
        "transformers": version("transformers"),
        "huggingface_hub": version("huggingface_hub"),
        "tokenizers": version("tokenizers"),
    }


def load_samples() -> list[dict[str, str]]:
    """Load NFC-normalised static text strings and pairing metadata."""
    with CORPUS_CSV.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_CORPUS_COLUMNS - columns
        if missing:
            raise ValueError(f"{CORPUS_CSV.name} missing columns: {sorted(missing)}")

        samples: list[dict[str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for line_number, row in enumerate(reader, start=2):
            sample = {key: (row.get(key) or "").strip() for key in REQUIRED_CORPUS_COLUMNS}
            if not all(sample[key] for key in REQUIRED_CORPUS_COLUMNS - {"adequacy_score"}):
                raise ValueError(f"Missing required value at {CORPUS_CSV.name}:{line_number}")
            sample["text"] = unicodedata.normalize("NFC", sample["text"])
            key = (sample["item_id"], sample["language"], sample["condition"])
            if key in seen:
                raise ValueError(f"Duplicate item/language/condition at {CORPUS_CSV.name}:{line_number}")
            seen.add(key)
            samples.append(sample)

    if not samples:
        raise ValueError(f"{CORPUS_CSV.name} contains no text samples")
    return samples


def utf8_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def measure(sample: dict[str, str], token_count: int) -> dict[str, object]:
    text = sample["text"]
    characters = len(text)
    bytes_ = utf8_bytes(text)
    return {
        "item_id": sample["item_id"],
        "category": sample["category"],
        "language": sample["language"],
        "condition": sample["condition"],
        "adequacy_score": sample["adequacy_score"],
        "text": text,
        "character_count": characters,
        "utf8_byte_count": bytes_,
        "token_count": token_count,
        "characters_per_token": characters / token_count,
        "bytes_per_token": bytes_ / token_count,
    }


def o200k_metadata(encoding: tiktoken.Encoding, versions: dict[str, str]) -> dict[str, object]:
    ranks = encoding._mergeable_ranks  # noqa: SLF001 - version fingerprint only
    fingerprint = hashlib.sha256(
        b"".join(token + rank.to_bytes(4, "big") for token, rank in sorted(ranks.items()))
    ).hexdigest()
    return {
        "name": "openai-o200k",
        "model_identifier": "o200k_base",
        "requested_revision": f"tiktoken-{versions['tiktoken']} (bundled encoding)",
        "resolved_revision": fingerprint,
        "library_versions": {"tiktoken": versions["tiktoken"]},
        "special_token_settings": {
            "encoding_call": "encode_ordinary(text)",
            "chat_template_used": False,
            "add_bos_token": False,
            "add_eos_token": False,
            "hidden_system_prompt_included": False,
        },
    }


def hf_metadata(name: str, spec: dict[str, str], tokenizer, versions: dict[str, str]) -> dict[str, object]:
    resolved_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    try:
        info = HfApi().model_info(spec["identifier"], revision=spec["revision"])
        resolved_revision = info.sha
    except Exception as error:  # Offline or gated access: retain requested revision.
        resolved_revision = resolved_revision or f"unresolved: {type(error).__name__}"

    special_map = {key: str(value) for key, value in tokenizer.special_tokens_map.items()}
    return {
        "name": name,
        "model_identifier": spec["identifier"],
        "requested_revision": spec["revision"],
        "resolved_revision": resolved_revision,
        "library_versions": {
            "transformers": versions["transformers"],
            "huggingface_hub": versions["huggingface_hub"],
            "tokenizers": versions["tokenizers"],
        },
        "special_token_settings": {
            "encoding_call": "encode(text, add_special_tokens=False)",
            "chat_template_used": False,
            "add_bos_token": False,
            "add_eos_token": False,
            "hidden_system_prompt_included": False,
            "tokenizer_add_bos_token": getattr(tokenizer, "add_bos_token", None),
            "tokenizer_add_eos_token": getattr(tokenizer, "add_eos_token", None),
            "special_tokens_map": special_map,
        },
    }


def main() -> None:
    versions = package_versions()
    samples = load_samples()
    results: dict[str, object] = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "NFC-normalised raw static strings only; no chat template, BOS, EOS, or system prompt.",
        "corpus": {
            "file": CORPUS_CSV.name,
            "sha256": hashlib.sha256(CORPUS_CSV.read_bytes()).hexdigest(),
            "sample_count": len(samples),
            "item_count": len({sample["item_id"] for sample in samples}),
        },
        "tokenizers": [],
        "failures": {},
    }
    rows: list[dict[str, object]] = []

    try:
        o200k = tiktoken.get_encoding("o200k_base")
        o200k_result = o200k_metadata(o200k, versions)
        o200k_result["measurements"] = [
            measure(sample, len(o200k.encode_ordinary(sample["text"]))) for sample in samples
        ]
        results["tokenizers"].append(o200k_result)
        rows.extend({"tokenizer": o200k_result["name"], **row} for row in o200k_result["measurements"])
    except Exception as error:
        results["failures"]["openai-o200k"] = f"{type(error).__name__}: {error}"

    for name, spec in HF_MODELS.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(spec["identifier"], revision=spec["revision"])
            tokenizer_result = hf_metadata(name, spec, tokenizer, versions)
            tokenizer_result["measurements"] = [
                measure(sample, len(tokenizer.encode(sample["text"], add_special_tokens=False)))
                for sample in samples
            ]
            results["tokenizers"].append(tokenizer_result)
            rows.extend({"tokenizer": tokenizer_result["name"], **row} for row in tokenizer_result["measurements"])
        except Exception as error:
            results["failures"][name] = f"{type(error).__name__}: {error}"

    OUTPUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "tokenizer", "item_id", "category", "language", "condition", "adequacy_score",
                "text", "character_count", "utf8_byte_count", "token_count",
                "characters_per_token", "bytes_per_token",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_JSON.name} and {OUTPUT_CSV.name}.")
    if results["failures"]:
        print("Failed tokenizers:", ", ".join(results["failures"]))


if __name__ == "__main__":
    main()
