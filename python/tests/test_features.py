"""Phase 1 tests: leakage, OOV/rare handling, hash determinism, time parsing.

These run on tiny in-memory frames. That is legitimate unit testing of the
transform logic; it is NOT fabricating benchmark data (no metrics come from here).
"""

import json

import polars as pl
import pytest

from pacer.data.features import (
    FeatureConfig,
    FeatureTransform,
    add_time_features,
    fnv1a_32,
)


def _frame(rows):
    return pl.DataFrame(rows)


# hour = YYMMDDHH. 14=2014. 2014-10-21 was a Tuesday (weekday 1).
BASE = {
    "hour": 14102108,
    "click": 0,
    "C1": "1005",
    "banner_pos": "0",
    "site_category": "cat_a",
    "app_category": "app_a",
    "device_type": "1",
    "device_conn_type": "2",
    "device_id": "dev1",
    "device_ip": "ip1",
    "site_id": "s1",
    "site_domain": "sd1",
    "app_id": "a1",
    "app_domain": "ad1",
    "device_model": "m1",
    "C14": "100",
    "C15": "320",
    "C16": "50",
    "C17": "1",
    "C18": "0",
    "C19": "35",
    "C20": "-1",
    "C21": "79",
}


def _small_config():
    # tiny bucket counts so tests are fast and collisions are checkable
    cfg = FeatureConfig(min_count=2)
    cfg.hash_fields = {k: 16 for k in cfg.hash_fields}
    return cfg


def test_fnv1a_known_vectors():
    # reference FNV-1a 32-bit values (well-known constants)
    assert fnv1a_32("") == 0x811C9DC5
    assert fnv1a_32("a") == 0xE40C292C
    assert fnv1a_32("foobar") == 0xBF9CF968


def test_hash_determinism():
    assert fnv1a_32("dev1") == fnv1a_32("dev1")
    t = FeatureTransform(_small_config()).fit(_frame([BASE, BASE]))
    a = t.transform(_frame([BASE]))
    b = t.transform(_frame([BASE]))
    assert a.to_dicts() == b.to_dicts()


def test_time_features_weekday():
    df = add_time_features(_frame([BASE]))
    assert df["hour_of_day"][0] == 8
    assert df["day_of_week"][0] == 1  # Tuesday


def test_oov_for_rare_and_unseen():
    # 'rareval' appears once (< min_count 2) -> OOV. 'common' appears twice -> kept.
    rows = [
        {**BASE, "site_category": "common"},
        {**BASE, "site_category": "common"},
        {**BASE, "site_category": "rareval"},
    ]
    t = FeatureTransform(_small_config()).fit(_frame(rows))
    out = t.transform(_frame(rows))
    idx = out["site_category"].to_list()
    # common -> some >0 index, rareval -> 0 (OOV)
    assert idx[0] == idx[1] > 0
    assert idx[2] == 0
    # a never-before-seen value also maps to OOV
    unseen = t.transform(_frame([{**BASE, "site_category": "brand_new"}]))
    assert unseen["site_category"][0] == 0


def test_no_train_val_leakage():
    """Vocab built on train must not gain entries from val values.

    A val-only category must land in OOV, proving the map was frozen on train.
    """
    train = [{**BASE, "site_category": f"c{i%3}"} for i in range(30)]
    val = [{**BASE, "site_category": "val_only"}]
    t = FeatureTransform(_small_config()).fit(_frame(train))
    train_vocab = set(t.vocab_maps["site_category"].keys())
    assert "val_only" not in train_vocab
    out = t.transform(_frame(val))
    assert out["site_category"][0] == 0  # OOV, not a fresh index


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        FeatureTransform(_small_config()).transform(_frame([BASE]))


def test_hash_bucket_range():
    cfg = _small_config()
    t = FeatureTransform(cfg).fit(_frame([BASE, BASE]))
    out = t.transform(_frame([BASE]))
    for fld, n in cfg.hash_fields.items():
        v = out[fld][0]
        assert 0 <= v < n
    # empty value -> OOV bucket 0
    empty = t.transform(_frame([{**BASE, "device_id": ""}]))
    assert empty["device_id"][0] == 0


def test_json_roundtrip_identical_transform(tmp_path):
    rows = [{**BASE, "site_category": "common"}, {**BASE, "site_category": "common"}]
    t = FeatureTransform(_small_config()).fit(_frame(rows))
    p = tmp_path / "transform.json"
    t.to_json(str(p))
    # artifact is valid json and reloads
    json.loads(p.read_text())
    t2 = FeatureTransform.from_json(str(p))
    a = t.transform(_frame([BASE]))
    b = t2.transform(_frame([BASE]))
    assert a.to_dicts() == b.to_dicts()
    assert t.field_order() == t2.field_order()


def test_cardinalities_include_oov():
    t = FeatureTransform(_small_config()).fit(_frame([BASE, BASE]))
    card = t.cardinalities()
    # one kept value + OOV = 2
    assert card["site_category"] == 2
    assert card["device_id"] == 16
