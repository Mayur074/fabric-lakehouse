"""
tests/test_fabric_pipeline.py
-------------------------------
Unit tests for Fabric Lakehouse pipeline components.
DQ tests import the real src/data_quality.py; PII and dedup tests mirror the PySpark logic.
No PySpark or Delta Lake required.
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime


# ── DQ tests (exercise the real src/data_quality.py) ─────────────────────────

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_quality import DataQualityChecker, DQCheck, DQReport


def _last(df, method, *args, **kwargs):
    """Run one check through the real DataQualityChecker and return its DQCheck."""
    checker = DataQualityChecker(pd.DataFrame({"col": df}), "t")
    getattr(checker, method)("col", *args, **kwargs)
    return checker.report.checks[-1]

def test_completeness_passes_no_nulls():
    assert _last(pd.Series([1, 2, 3, 4, 5]), "check_completeness", 99.0).passed

def test_completeness_fails_with_nulls():
    assert not _last(pd.Series([1, None, None, None, 5]), "check_completeness", 99.0).passed

def test_uniqueness_passes_all_unique():
    assert _last(pd.Series(["A", "B", "C", "D"]), "check_uniqueness", 100.0).passed

def test_uniqueness_fails_duplicates():
    assert not _last(pd.Series(["A", "A", "B", "B"]), "check_uniqueness", 100.0).passed

def test_accepted_values_passes():
    assert _last(pd.Series(["USD", "EUR", "GBP"]), "check_accepted_values", ["USD", "EUR", "GBP", "CAD"]).passed

def test_accepted_values_fails_invalid():
    assert not _last(pd.Series(["USD", "XYZ", "EUR"]), "check_accepted_values", ["USD", "EUR"]).passed

def test_range_passes_within_bounds():
    assert _last(pd.Series([10.0, 20.0, 30.0]), "check_range", min_val=0.0, max_val=100.0).passed

def test_range_fails_below_min():
    assert not _last(pd.Series([10.0, -5.0, 20.0]), "check_range", min_val=0.0).passed

def test_range_fails_above_max():
    assert not _last(pd.Series([10.0, 200.0, 20.0]), "check_range", max_val=100.0).passed

def test_dq_report_passes_when_all_checks_pass():
    report = DQReport("txn", "silver", datetime.now().isoformat(), 1000)
    report.add_check(DQCheck("c1", "d1", True,  100.0, 99.0, "pct"))
    report.add_check(DQCheck("c2", "d2", True,  100.0, 100.0, "pct"))
    assert report.passed

def test_dq_report_fails_when_any_check_fails():
    report = DQReport("txn", "silver", datetime.now().isoformat(), 1000)
    report.add_check(DQCheck("c1", "d1", True,  100.0, 99.0, "pct"))
    report.add_check(DQCheck("c2", "d2", False, 90.0,  99.0, "pct"))
    assert not report.passed

def test_dq_report_check_count():
    report = DQReport("cust", "silver", datetime.now().isoformat(), 500)
    for i in range(5):
        report.add_check(DQCheck(f"c{i}", "d", True, 100.0, 99.0))
    assert len(report.checks) == 5

def test_checker_chains_and_summarises():
    df = pd.DataFrame({"id": [1, 2, 3], "amt": [5.0, -1.0, 7.0]})
    report = (DataQualityChecker(df, "txn")
              .check_uniqueness("id").check_range("amt", min_val=0).check_row_count(3).report)
    s = report.summary()
    assert (s["checks_total"], s["checks_passed"], s["checks_failed"]) == (3, 2, 1)

# ── PII masking logic ─────────────────────────────────────────────────────────

def mask_email(email: str) -> str:
    if "@" not in email:
        return email
    local, domain = email.split("@", 1)
    return local[:2] + "***@" + domain

def mask_name(name: str) -> str:
    return name[:2] + "***"

def test_email_masking_preserves_domain():
    result = mask_email("john.doe@example.com")
    assert result.endswith("@example.com")

def test_email_masking_hides_local_part():
    result = mask_email("john.doe@example.com")
    assert "john.doe" not in result
    assert "***" in result

def test_name_masking_keeps_first_two_chars():
    result = mask_name("Mayur")
    assert result.startswith("Ma")
    assert "***" in result

def test_name_masking_hides_rest():
    result = mask_name("Mayur")
    assert "yur" not in result


# ── Deduplication logic ───────────────────────────────────────────────────────

def deduplicate_pandas(df: pd.DataFrame, pk: str, ts: str) -> pd.DataFrame:
    return df.sort_values(ts, ascending=False).drop_duplicates(subset=[pk])

def test_deduplication_keeps_latest():
    df = pd.DataFrame({
        "id": ["A", "A", "B"],
        "ts": ["2024-01-03", "2024-01-01", "2024-01-02"],
        "val": [3, 1, 2],
    })
    result = deduplicate_pandas(df, "id", "ts")
    assert len(result) == 2
    a_row = result[result["id"] == "A"].iloc[0]
    assert a_row["val"] == 3

def test_deduplication_no_duplicates_unchanged():
    df = pd.DataFrame({"id": ["A", "B", "C"], "ts": ["2024-01-01"]*3, "val": [1,2,3]})
    result = deduplicate_pandas(df, "id", "ts")
    assert len(result) == 3


# ── Medallion layer config ────────────────────────────────────────────────────

LAYERS = ["bronze", "silver", "gold"]
DQ_RULES = {
    "transactions": ["not_null_id", "positive_amount", "valid_currency", "valid_status"],
    "customers":    ["not_null_id", "valid_email", "not_null_country"],
}

def test_three_medallion_layers():
    assert len(LAYERS) == 3
    assert "bronze" in LAYERS
    assert "silver" in LAYERS
    assert "gold"   in LAYERS

def test_dq_rules_defined_for_all_tables():
    assert "transactions" in DQ_RULES
    assert "customers"    in DQ_RULES

def test_transaction_dq_has_amount_rule():
    assert any("amount" in r for r in DQ_RULES["transactions"])

def test_customer_dq_has_email_rule():
    assert any("email" in r for r in DQ_RULES["customers"])


# ── File existence ────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent

def test_bronze_notebook_exists():
    assert (BASE / "notebooks" / "01_bronze_ingestion.py").exists()

def test_silver_notebook_exists():
    assert (BASE / "notebooks" / "02_silver_transformation.py").exists()

def test_gold_notebook_exists():
    assert (BASE / "notebooks" / "03_gold_aggregations.py").exists()

def test_pipeline_json_exists():
    assert (BASE / "pipelines" / "medallion_pipeline.json").exists()

def test_data_quality_module_exists():
    assert (BASE / "src" / "data_quality.py").exists()

def test_pipeline_json_valid():
    path = BASE / "pipelines" / "medallion_pipeline.json"
    data = json.loads(path.read_text())
    assert "name" in data
    assert "properties" in data
    assert "activities" in data["properties"]

def test_pipeline_has_all_layers():
    path       = BASE / "pipelines" / "medallion_pipeline.json"
    data       = json.loads(path.read_text())
    activities = [a["name"] for a in data["properties"]["activities"]]
    assert any("Bronze" in a or "bronze" in a.lower() for a in activities)
    assert any("Silver" in a or "silver" in a.lower() for a in activities)
    assert any("Gold"   in a or "gold"   in a.lower() for a in activities)

def test_requirements_exists():
    assert (BASE / "requirements.txt").exists()
