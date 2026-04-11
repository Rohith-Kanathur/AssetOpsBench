"""Shared text normalization and slug helpers for the scenario pipeline."""

from __future__ import annotations

import re


def slugify_asset_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "asset"


def collapse_whitespace_lower(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_for_fuzzy_dedup(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def normalize_example_fingerprint(text: str) -> str:
    return " ".join(str(text).lower().split())


__all__ = [
    "collapse_whitespace_lower",
    "normalize_example_fingerprint",
    "normalize_for_fuzzy_dedup",
    "slugify_asset_name",
]
