"""Feature transform for Avazu CTR data.

Design goals
------------
1. Fit statistics (vocabularies, rare-value counts) on the TRAIN split only, then
   apply the frozen transform to val/test. This is the leakage guarantee.
2. Serialize the fitted transform to JSON so the Go serving layer can apply the
   *identical* transform without reimplementing any logic by hand.

Two encoding paths
------------------
* ``vocab`` fields (low cardinality): build a value -> index map on train. Values
  seen fewer than ``min_count`` times, and any value unseen at transform time, map
  to a shared OOV bucket (index 0). The map is small and cheap to serialize.

* ``hash`` fields (high cardinality): hash the raw string into a fixed bucket
  count with a deterministic FNV-1a hash that Go can reproduce byte-for-byte. We
  deliberately do NOT keep a per-value rare-count map for hashed fields: that
  would rebuild the very vocabulary hashing exists to avoid, and would bloat the
  serialized artifact (device_ip alone has millions of values). Hashing already
  absorbs rare/unseen values gracefully via collisions. Bucket 0 is reserved so a
  missing/empty value maps there. This tradeoff is documented in the README.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

import polars as pl

# FNV-1a 32-bit constants. Chosen because it is trivial to reimplement identically
# in Go, unlike polars' internal ahash/xxhash which we could not match cross-language.
_FNV_OFFSET_32 = 0x811C9DC5
_FNV_PRIME_32 = 0x01000193
_MASK_32 = 0xFFFFFFFF


def fnv1a_32(s: str) -> int:
    """Deterministic FNV-1a hash over the UTF-8 bytes of ``s``.

    Must stay byte-for-byte identical to the Go implementation in
    go/internal/model. Do not "optimize" this without updating both sides and the
    parity test.
    """
    h = _FNV_OFFSET_32
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * _FNV_PRIME_32) & _MASK_32
    return h


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Raw Avazu columns. ``id`` is dropped, ``click`` is the label, ``hour`` is
# expanded into day_of_week / hour_of_day before encoding.
LABEL_COL = "click"
DROP_COLS = ["id"]
TIME_COL = "hour"

# Default field -> encoding spec. Bucket counts are config values, not magic
# numbers buried in code. Widest hashing (2^20) for the two truly enormous-
# cardinality identifier fields; 2^16 for the rest of the high-card fields.
HASH_WIDE = 1 << 20
HASH_MID = 1 << 16


@dataclass
class FeatureConfig:
    """Which columns get which treatment. Fully declarative + serializable."""

    # vocab fields: value -> dense index, with OOV + rare bucketing
    vocab_fields: list[str] = field(
        default_factory=lambda: [
            "C1",
            "banner_pos",
            "site_category",
            "app_category",
            "device_type",
            "device_conn_type",
            "day_of_week",
            "hour_of_day",
        ]
    )
    # hashed fields: field -> bucket count
    hash_fields: dict[str, int] = field(
        default_factory=lambda: {
            "device_id": HASH_WIDE,
            "device_ip": HASH_WIDE,
            "site_id": HASH_MID,
            "site_domain": HASH_MID,
            "app_id": HASH_MID,
            "app_domain": HASH_MID,
            "device_model": HASH_MID,
            "C14": HASH_MID,
            "C15": HASH_MID,
            "C16": HASH_MID,
            "C17": HASH_MID,
            "C18": HASH_MID,
            "C19": HASH_MID,
            "C20": HASH_MID,
            "C21": HASH_MID,
        }
    )
    min_count: int = 10  # vocab values rarer than this collapse to OOV

    def to_dict(self) -> dict:
        return {
            "vocab_fields": self.vocab_fields,
            "hash_fields": self.hash_fields,
            "min_count": self.min_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        return cls(
            vocab_fields=list(d["vocab_fields"]),
            hash_fields={k: int(v) for k, v in d["hash_fields"].items()},
            min_count=int(d["min_count"]),
        )


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """Expand Avazu ``hour`` (YYMMDDHH int) into day_of_week + hour_of_day.

    day_of_week is 0=Monday..6=Sunday, matching python date.weekday().
    """
    s = df[TIME_COL].cast(pl.Utf8).str.zfill(8)
    yy = s.str.slice(0, 2).cast(pl.Int32)
    mm = s.str.slice(2, 2).cast(pl.Int32)
    dd = s.str.slice(4, 2).cast(pl.Int32)
    hh = s.str.slice(6, 2).cast(pl.Int32)

    # Build a real date to get weekday; Avazu is year 2014.
    dow = pl.Series(
        "day_of_week",
        [
            date(2000 + y, m, d).weekday()
            for y, m, d in zip(yy.to_list(), mm.to_list(), dd.to_list())
        ],
        dtype=pl.Int32,
    )
    return df.with_columns(
        [
            hh.alias("hour_of_day"),
            dow,
        ]
    )


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------

# Index 0 is reserved as OOV / missing across every field.
OOV_INDEX = 0


@dataclass
class FeatureTransform:
    config: FeatureConfig
    # field -> {raw_value_str: dense_index}. index 0 reserved for OOV.
    vocab_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    fitted: bool = False

    # -- fitting -----------------------------------------------------------
    def fit(self, train_df: pl.DataFrame) -> "FeatureTransform":
        """Compute vocab maps on TRAIN ONLY. Raises if given non-train data by
        accident is impossible to detect, so leakage is enforced by the caller +
        tests, not here."""
        df = add_time_features(train_df)
        self.vocab_maps = {}
        for col in self.config.vocab_fields:
            counts = (
                df.select(pl.col(col).cast(pl.Utf8).alias("v"))
                .group_by("v")
                .len()
            )
            # keep only values at/above min_count; everything else -> OOV
            kept = counts.filter(pl.col("len") >= self.config.min_count)
            # deterministic ordering for reproducible indices
            values = sorted(kept["v"].to_list())
            self.vocab_maps[col] = {v: i + 1 for i, v in enumerate(values)}
        self.fitted = True
        return self

    # -- applying ----------------------------------------------------------
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeatureTransform.transform called before fit")
        df = add_time_features(df)
        out_cols: list[pl.Series] = []

        for col in self.config.vocab_fields:
            vmap = self.vocab_maps[col]
            vals = df[col].cast(pl.Utf8).to_list()
            idx = [vmap.get(v, OOV_INDEX) for v in vals]
            out_cols.append(pl.Series(col, idx, dtype=pl.Int64))

        for col, n_buckets in self.config.hash_fields.items():
            vals = df[col].cast(pl.Utf8).to_list()
            # bucket 0 reserved for empty/missing; real values -> [1, n_buckets)
            idx = [
                OOV_INDEX
                if v is None or v == ""
                else 1 + (fnv1a_32(v) % (n_buckets - 1))
                for v in vals
            ]
            out_cols.append(pl.Series(col, idx, dtype=pl.Int64))

        result = pl.DataFrame(out_cols)
        if LABEL_COL in df.columns:
            result = result.with_columns(df[LABEL_COL].cast(pl.Int8).alias(LABEL_COL))
        return result

    # -- cardinalities -----------------------------------------------------
    def cardinalities(self) -> dict[str, int]:
        """Number of distinct indices each field can emit (including OOV)."""
        card: dict[str, int] = {}
        for col in self.config.vocab_fields:
            card[col] = len(self.vocab_maps[col]) + 1  # + OOV
        for col, n_buckets in self.config.hash_fields.items():
            card[col] = n_buckets
        return card

    def field_order(self) -> list[str]:
        """Stable column order the model + Go both rely on."""
        return list(self.config.vocab_fields) + list(self.config.hash_fields.keys())

    # -- serialization -----------------------------------------------------
    def to_json(self, path: str) -> None:
        payload = {
            "hash": {"algo": "fnv1a_32", "offset": _FNV_OFFSET_32, "prime": _FNV_PRIME_32},
            "config": self.config.to_dict(),
            "vocab_maps": self.vocab_maps,
            "field_order": self.field_order(),
            "oov_index": OOV_INDEX,
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def from_json(cls, path: str) -> "FeatureTransform":
        with open(path) as f:
            payload = json.load(f)
        cfg = FeatureConfig.from_dict(payload["config"])
        t = cls(config=cfg, vocab_maps=payload["vocab_maps"], fitted=True)
        return t
