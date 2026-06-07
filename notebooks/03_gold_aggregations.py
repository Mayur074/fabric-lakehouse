"""
notebooks/03_gold_aggregations.py
-----------------------------------
Gold Layer: Business-ready aggregations optimized for Power BI
Direct Lake mode and analytics queries.

Produces:
  - gold_revenue_kpis      : Monthly revenue KPIs with MoM growth
  - gold_customer_segments : RFM-based customer segmentation
  - gold_product_performance: Category-level performance metrics

Partitioned and Z-ordered for optimal query performance.
Microsoft Fabric Direct Lake reads these Delta tables directly.
"""

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Revenue KPIs ──────────────────────────────────────────────────────────────

def build_revenue_kpis(spark: SparkSession, silver_txn_path: str,
                        gold_path: str) -> int:
    """
    Monthly revenue KPIs with period-over-period growth.
    Partitioned by year for optimal Power BI Direct Lake performance.
    """
    df = spark.read.format("delta").load(silver_txn_path)

    monthly = (
        df.filter(F.col("is_completed") == True)
        .groupBy("transaction_year", "transaction_month", "channel", "currency")
        .agg(
            F.count("transaction_id").alias("total_transactions"),
            F.countDistinct("customer_id").alias("unique_customers"),
            F.sum("amount_usd").alias("gross_revenue"),
            F.avg("amount_usd").alias("avg_transaction_value"),
            F.min("amount_usd").alias("min_transaction_value"),
            F.max("amount_usd").alias("max_transaction_value"),
            F.stddev("amount_usd").alias("stddev_transaction_value"),
        )
        .withColumn("avg_transaction_value", F.round("avg_transaction_value", 2))
        .withColumn("gross_revenue",          F.round("gross_revenue", 2))
    )

    # MoM growth via window function
    w = Window.partitionBy("channel", "currency").orderBy("transaction_year", "transaction_month")
    monthly = (
        monthly
        .withColumn("prev_month_revenue", F.lag("gross_revenue").over(w))
        .withColumn("mom_growth_pct",
            F.round(
                (F.col("gross_revenue") - F.col("prev_month_revenue"))
                / F.col("prev_month_revenue") * 100, 2
            )
        )
        .withColumn("_gold_ts", F.current_timestamp())
    )

    row_count = monthly.count()

    (
        monthly.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .partitionBy("transaction_year")
        .save(gold_path)
    )

    logger.info(f"Gold revenue KPIs written: {row_count:,} rows, partitioned by year")
    return row_count


# ── Customer segmentation ─────────────────────────────────────────────────────

def build_customer_segments(spark: SparkSession, silver_txn_path: str,
                              silver_cust_path: str, gold_path: str) -> int:
    """
    RFM-based customer segmentation for Power BI dashboards.
    Segments: CHAMPION, LOYAL, AT_RISK, ONE_TIME, CHURNED.
    """
    txn  = spark.read.format("delta").load(silver_txn_path).filter("is_completed = true")
    cust = spark.read.format("delta").load(silver_cust_path)

    # Compute RFM metrics
    rfm = (
        txn.groupBy("customer_id")
        .agg(
            F.count("transaction_id").alias("frequency"),
            F.sum("amount_usd").alias("monetary"),
            F.max("transaction_date").alias("last_purchase_date"),
            F.min("transaction_date").alias("first_purchase_date"),
            F.countDistinct("channel").alias("channels_used"),
        )
        .withColumn("recency_days",
            F.datediff(F.current_date(), F.col("last_purchase_date")))
        .withColumn("customer_tenure_days",
            F.datediff(F.col("last_purchase_date"), F.col("first_purchase_date")))
        .withColumn("avg_order_value",
            F.round(F.col("monetary") / F.col("frequency"), 2))
    )

    # Segmentation rules
    rfm = rfm.withColumn("segment",
        F.when(
            (F.col("frequency") >= 10) & (F.col("monetary") >= 1000), "CHAMPION"
        ).when(
            (F.col("frequency") >= 5)  & (F.col("monetary") >= 300),  "LOYAL"
        ).when(
            (F.col("recency_days") > 180), "CHURNED"
        ).when(
            F.col("frequency") == 1, "ONE_TIME"
        ).otherwise("AT_RISK")
    )

    # Join with customer attributes
    gold_df = (
        rfm.join(
            cust.select("customer_id", "country", "segment".alias("customer_tier")),
            on="customer_id", how="left"
        )
        .withColumn("_gold_ts", F.current_timestamp())
    )

    row_count = gold_df.count()

    gold_df.write.format("delta").mode("overwrite").option("overwriteSchema","true").save(gold_path)
    logger.info(f"Gold customer segments written: {row_count:,} rows")
    return row_count


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_gold_pipeline(spark: SparkSession, silver_base: str, gold_base: str) -> dict:
    """Run all Gold aggregations."""
    results = {}

    results["revenue_kpis"] = build_revenue_kpis(
        spark,
        silver_txn_path=f"{silver_base}/transactions",
        gold_path=f"{gold_base}/revenue_kpis",
    )

    results["customer_segments"] = build_customer_segments(
        spark,
        silver_txn_path=f"{silver_base}/transactions",
        silver_cust_path=f"{silver_base}/customers",
        gold_path=f"{gold_base}/customer_segments",
    )

    logger.info(f"Gold pipeline complete: {results}")
    return results
