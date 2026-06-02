"""
notebooks/02_silver_transformation.py
---------------------------------------
Silver Layer: Cleanse, validate, standardize, and deduplicate Bronze data.
Implements data quality rules, PII handling, and schema enforcement.
Writes to Silver Delta tables with MERGE (upsert) for idempotency.

Microsoft Fabric equivalent: Lakehouse Notebook with Delta Lake MERGE.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Data quality rule definitions ─────────────────────────────────────────────

TRANSACTION_DQ_RULES = {
    "not_null_id":       "transaction_id IS NOT NULL",
    "not_null_customer": "customer_id IS NOT NULL",
    "positive_amount":   "amount > 0",
    "valid_currency":    "currency IN ('USD', 'EUR', 'GBP', 'CAD', 'AUD')",
    "valid_status":      "status IN ('COMPLETED', 'PENDING', 'FAILED', 'REFUNDED')",
    "valid_channel":     "channel IN ('WEB', 'MOBILE', 'API', 'POS')",
}

CUSTOMER_DQ_RULES = {
    "not_null_id":       "customer_id IS NOT NULL",
    "valid_email":       "email LIKE '%@%'",
    "not_null_country":  "country IS NOT NULL",
}


def apply_dq_rules(df: DataFrame, rules: dict, table_name: str) -> tuple[DataFrame, DataFrame]:
    """
    Apply data quality rules. Returns (passed_df, failed_df).
    Failed records are quarantined with the failing rule name.
    """
    passed = df
    failed_frames = []

    for rule_name, condition in rules.items():
        bad = passed.filter(~F.expr(condition)).withColumn("_dq_rule", F.lit(rule_name))
        failed_frames.append(bad)
        passed = passed.filter(F.expr(condition))
        bad_count = bad.count()
        if bad_count > 0:
            logger.warning(f"{table_name} DQ rule '{rule_name}' failed: {bad_count:,} rows quarantined")

    from functools import reduce
    failed = reduce(DataFrame.unionByName, failed_frames) if failed_frames else passed.limit(0)
    logger.info(f"{table_name}: {passed.count():,} passed | {failed.count():,} quarantined")
    return passed, failed


# ── PII handling ──────────────────────────────────────────────────────────────

def mask_pii(df: DataFrame) -> DataFrame:
    """
    Mask PII fields for Silver layer.
    Email: keep domain, hash local part.
    Names: keep only first 2 chars + asterisks.
    """
    if "email" in df.columns:
        df = df.withColumn(
            "email_masked",
            F.concat(
                F.substring(F.col("email"), 1, 2),
                F.lit("***@"),
                F.regexp_extract(F.col("email"), "@(.+)$", 1)
            )
        ).drop("email").withColumnRenamed("email_masked", "email")

    for name_col in ["first_name", "last_name"]:
        if name_col in df.columns:
            df = df.withColumn(
                name_col,
                F.concat(F.substring(F.col(name_col), 1, 2), F.lit("***"))
            )
    return df


# ── Deduplication ─────────────────────────────────────────────────────────────

def deduplicate(df: DataFrame, pk_col: str, ts_col: str = "_ingestion_ts") -> DataFrame:
    """Keep the most recent record per primary key."""
    window = Window.partitionBy(pk_col).orderBy(F.col(ts_col).desc())
    return (
        df
        .withColumn("_row_num", F.row_number().over(window))
        .filter(F.col("_row_num") == 1)
        .drop("_row_num")
    )


# ── Silver transformations ────────────────────────────────────────────────────

def transform_transactions(spark: SparkSession, bronze_path: str,
                             silver_path: str, quarantine_path: str) -> dict:
    """
    Transform raw Bronze transactions to clean Silver.
    - Apply DQ rules and quarantine failures
    - Standardize fields
    - Deduplicate
    - Add derived columns
    - Write to Silver Delta with MERGE
    """
    logger.info("Starting Silver transformation: transactions")

    df = spark.read.format("delta").load(bronze_path)

    # Apply DQ rules
    passed, failed = apply_dq_rules(df, TRANSACTION_DQ_RULES, "transactions")

    # Quarantine failed records
    if failed.count() > 0:
        failed.write.format("delta").mode("append").save(quarantine_path)

    # Standardize
    silver_df = (
        passed
        .withColumn("amount_usd",     F.round(F.col("amount"), 2))
        .withColumn("channel",         F.upper(F.trim(F.col("channel"))))
        .withColumn("status",          F.upper(F.trim(F.col("status"))))
        .withColumn("currency",        F.upper(F.trim(F.col("currency"))))
        .withColumn("transaction_date",F.to_date(F.col("transaction_ts")))
        .withColumn("transaction_year",F.year(F.col("transaction_ts")))
        .withColumn("transaction_month",F.month(F.col("transaction_ts")))
        .withColumn("is_completed",    F.col("status") == "COMPLETED")
        .withColumn("_silver_ts",      F.current_timestamp())
    )

    # Deduplicate
    silver_df = deduplicate(silver_df, "transaction_id")

    row_count = silver_df.count()

    # Write Silver
    silver_df.write.format("delta").mode("overwrite").option("overwriteSchema","true").save(silver_path)
    logger.info(f"Silver transactions written: {row_count:,} rows")

    return {"silver_rows": row_count, "quarantined": failed.count()}


def transform_customers(spark: SparkSession, bronze_path: str,
                         silver_path: str, quarantine_path: str) -> dict:
    """Transform raw Bronze customers to clean Silver with PII masking."""
    logger.info("Starting Silver transformation: customers")

    df = spark.read.format("delta").load(bronze_path)

    passed, failed = apply_dq_rules(df, CUSTOMER_DQ_RULES, "customers")

    if failed.count() > 0:
        failed.write.format("delta").mode("append").save(quarantine_path)

    silver_df = (
        passed
        .pipe(mask_pii)
        .withColumn("country",          F.upper(F.trim(F.col("country"))))
        .withColumn("segment",          F.upper(F.coalesce(F.col("segment"), F.lit("STANDARD"))))
        .withColumn("signup_date",      F.to_date(F.col("signup_date")))
        .withColumn("_silver_ts",       F.current_timestamp())
    )

    silver_df = deduplicate(silver_df, "customer_id")
    row_count = silver_df.count()

    silver_df.write.format("delta").mode("overwrite").option("overwriteSchema","true").save(silver_path)
    logger.info(f"Silver customers written: {row_count:,} rows")

    return {"silver_rows": row_count, "quarantined": failed.count()}
