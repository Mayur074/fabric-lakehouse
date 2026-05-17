"""
src/data_quality.py
--------------------
Reusable data quality framework for Medallion pipeline.
Implements Great Expectations-style checks without the dependency.
Tracks DQ results across pipeline runs for observability.
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DQ_LOG_PATH = "data/dq_results.json"


@dataclass
class DQCheck:
    name:        str
    description: str
    passed:      bool
    value:       float
    threshold:   float
    unit:        str = "count"


@dataclass
class DQReport:
    table_name:  str
    layer:       str
    run_ts:      str
    total_rows:  int
    checks:      list[DQCheck] = field(default_factory=list)
    passed:      bool = True

    def add_check(self, check: DQCheck):
        self.checks.append(check)
        if not check.passed:
            self.passed = False
            logger.warning(f"DQ FAIL [{self.table_name}] {check.name}: "
                           f"{check.value} (threshold: {check.threshold})")
        else:
            logger.info(f"DQ PASS [{self.table_name}] {check.name}")

    def summary(self) -> dict:
        n_pass = sum(1 for c in self.checks if c.passed)
        return {
            "table":      self.table_name,
            "layer":      self.layer,
            "run_ts":     self.run_ts,
            "total_rows": self.total_rows,
            "checks_total":  len(self.checks),
            "checks_passed": n_pass,
            "checks_failed": len(self.checks) - n_pass,
            "overall_passed": self.passed,
        }

    def save(self, path: str = DQ_LOG_PATH):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        log = json.loads(Path(path).read_text()) if Path(path).exists() else []
        log.append(asdict(self))
        Path(path).write_text(json.dumps(log, indent=2))


# ── Check library ─────────────────────────────────────────────────────────────

class DataQualityChecker:
    """Run data quality checks on a pandas DataFrame."""

    def __init__(self, df: pd.DataFrame, table_name: str, layer: str = "silver"):
        self.df    = df
        self.report= DQReport(
            table_name=table_name,
            layer=layer,
            run_ts=datetime.now().isoformat(),
            total_rows=len(df),
        )

    def check_completeness(self, column: str, threshold_pct: float = 99.0) -> "DataQualityChecker":
        """Column must have >= threshold_pct non-null values."""
        pct   = (1 - self.df[column].isna().mean()) * 100
        check = DQCheck(
            name=f"completeness_{column}",
            description=f"{column} non-null >= {threshold_pct}%",
            passed=pct >= threshold_pct,
            value=round(pct, 2), threshold=threshold_pct, unit="pct",
        )
        self.report.add_check(check)
        return self

    def check_uniqueness(self, column: str, threshold_pct: float = 100.0) -> "DataQualityChecker":
        """Column must have >= threshold_pct unique values."""
        pct   = self.df[column].nunique() / len(self.df) * 100
        check = DQCheck(
            name=f"uniqueness_{column}",
            description=f"{column} unique >= {threshold_pct}%",
            passed=pct >= threshold_pct,
            value=round(pct, 2), threshold=threshold_pct, unit="pct",
        )
        self.report.add_check(check)
        return self

    def check_accepted_values(self, column: str, values: list) -> "DataQualityChecker":
        """All non-null column values must be in accepted set."""
        invalid = (~self.df[column].isin(values) & self.df[column].notna()).sum()
        check   = DQCheck(
            name=f"accepted_values_{column}",
            description=f"{column} in {values}",
            passed=invalid == 0,
            value=float(invalid), threshold=0, unit="count",
        )
        self.report.add_check(check)
        return self

    def check_range(self, column: str, min_val: float = None,
                    max_val: float = None) -> "DataQualityChecker":
        """Numeric column values must be within [min_val, max_val]."""
        col = self.df[column].dropna()
        out = 0
        if min_val is not None:
            out += (col < min_val).sum()
        if max_val is not None:
            out += (col > max_val).sum()
        check = DQCheck(
            name=f"range_{column}",
            description=f"{column} in [{min_val}, {max_val}]",
            passed=out == 0,
            value=float(out), threshold=0, unit="count",
        )
        self.report.add_check(check)
        return self

    def check_row_count(self, min_rows: int) -> "DataQualityChecker":
        """Table must have at least min_rows rows."""
        check = DQCheck(
            name="row_count",
            description=f"row count >= {min_rows}",
            passed=len(self.df) >= min_rows,
            value=float(len(self.df)), threshold=float(min_rows), unit="count",
        )
        self.report.add_check(check)
        return self

    def run(self) -> DQReport:
        self.report.save()
        logger.info(f"DQ Report: {self.report.summary()}")
        return self.report
