"""
scripts/generate_and_run.py
-----------------------------
Generates synthetic e-commerce data and runs the full
Bronze → Silver → Gold Medallion pipeline locally using PySpark + Delta Lake.

In Microsoft Fabric: each notebook runs as a Fabric Spark job.
Locally: this script orchestrates the full pipeline end-to-end.
"""

from __future__ import annotations
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import random

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

# Paths
BASE        = Path("data/lakehouse")
RAW_PATH    = BASE / "raw"
BRONZE_PATH = str(BASE / "bronze")
SILVER_PATH = str(BASE / "silver")
GOLD_PATH   = str(BASE / "gold")
QUARANTINE  = str(BASE / "quarantine")

N_CUSTOMERS    = 2_000
N_TRANSACTIONS = 30_000


def generate_customers(n: int) -> list[dict]:
    countries  = ["US", "UK", "DE", "FR", "CA", "AU"]
    segments   = ["STANDARD", "PREMIUM", "VIP"]
    weights    = [0.60, 0.30, 0.10]
    start      = datetime(2019, 1, 1)

    return [
        {
            "customer_id": f"C{i:05d}",
            "first_name":  f"First{i}",
            "last_name":   f"Last{i}",
            "email":       f"user{i}@example.com",
            "country":     random.choice(countries),
            "segment":     random.choices(segments, weights)[0],
            "signup_date": str((start + timedelta(days=random.randint(0, 1800))).date()),
            "_ingestion_ts": datetime.now().isoformat(),
            "_source":     "crm_system",
        }
        for i in range(1, n + 1)
    ]


def generate_transactions(n: int, n_customers: int) -> list[dict]:
    statuses  = ["COMPLETED", "PENDING", "FAILED", "REFUNDED"]
    s_weights = [0.75, 0.10, 0.08, 0.07]
    channels  = ["WEB", "MOBILE", "API", "POS"]
    c_weights = [0.40, 0.35, 0.15, 0.10]
    currencies= ["USD", "EUR", "GBP", "CAD"]
    start     = datetime(2021, 1, 1)

    return [
        {
            "transaction_id":  f"T{i:08d}",
            "customer_id":     f"C{random.randint(1, n_customers):05d}",
            "product_id":      f"P{random.randint(1, 500):04d}",
            "transaction_ts":  (start + timedelta(
                days=random.randint(0, 1460),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )).isoformat(),
            "amount":          round(np.random.lognormal(4.0, 1.2), 2),
            "currency":        random.choices(currencies, [0.70,0.15,0.10,0.05])[0],
            "channel":         random.choices(channels, c_weights)[0],
            "status":          random.choices(statuses, s_weights)[0],
            "_ingestion_ts":   datetime.now().isoformat(),
            "_source":         "payment_gateway",
        }
        for i in range(1, n + 1)
    ]


def setup_raw_data():
    RAW_PATH.mkdir(parents=True, exist_ok=True)
    logger.info("Generating synthetic data...")

    customers    = generate_customers(N_CUSTOMERS)
    transactions = generate_transactions(N_TRANSACTIONS, N_CUSTOMERS)

    # Save customers as CSV
    pd.DataFrame(customers).to_csv(RAW_PATH / "customers.csv", index=False)

    # Save transactions as JSON (chunked for realistic ingestion)
    chunk_size = 5_000
    for i, start in enumerate(range(0, len(transactions), chunk_size)):
        chunk = transactions[start:start + chunk_size]
        (RAW_PATH / f"transactions_{i:02d}.json").write_text(
            json.dumps(chunk, indent=2)
        )

    logger.info(f"Raw data written to {RAW_PATH}")
    logger.info(f"  Customers:    {len(customers):,}")
    logger.info(f"  Transactions: {len(transactions):,}")


if __name__ == "__main__":
    setup_raw_data()
    logger.info("\nRaw data ready. To run the full pipeline:")
    logger.info("  pip install pyspark delta-spark")
    logger.info("  python -c \"")
    logger.info("    from scripts.generate_and_run import *")
    logger.info("    from notebooks.01_bronze_ingestion import *")
    logger.info("    spark = get_spark()")
    logger.info("    ingest_transactions_batch(spark, 'data/lakehouse/raw', 'data/lakehouse/bronze/transactions')")
    logger.info("  \"")
    logger.info("\nOr deploy notebooks to Microsoft Fabric Lakehouse directly.")
