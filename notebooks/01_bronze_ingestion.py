"""
notebooks/01_bronze_ingestion.py
---------------------------------
Bronze Layer: Raw data ingestion into OneLake / ADLS Gen2.
Simulates Microsoft Fabric Lakehouse ingestion patterns using
PySpark (runs in Fabric Spark runtime or locally via pyspark).

Ingests:
  - Streaming events via Event Hub simulation (batch mode fallback)
  - CSV/JSON files from source systems
  - API responses (REST)

Writes raw, unmodified data to Bronze Delta tables.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, TimestampType, LongType
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Spark session (local mode for dev; Fabric uses managed session) ────────────
def get_spark(app_name: str = "FabricBronze") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .master("local[*]")
        .getOrCreate()
    )


# ── Schemas ───────────────────────────────────────────────────────────────────

TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id",  StringType(),    False),
    StructField("customer_id",     StringType(),    False),
    StructField("product_id",      StringType(),    False),
    StructField("transaction_ts",  TimestampType(), True),
    StructField("amount",          DoubleType(),    True),
    StructField("currency",        StringType(),    True),
    StructField("channel",         StringType(),    True),
    StructField("status",          StringType(),    True),
    StructField("_ingestion_ts",   TimestampType(), True),
    StructField("_source",         StringType(),    True),
])

CUSTOMER_SCHEMA = StructType([
    StructField("customer_id",   StringType(),  False),
    StructField("first_name",    StringType(),  True),
    StructField("last_name",     StringType(),  True),
    StructField("email",         StringType(),  True),
    StructField("country",       StringType(),  True),
    StructField("segment",       StringType(),  True),
    StructField("signup_date",   StringType(),  True),
    StructField("_ingestion_ts", TimestampType(),True),
    StructField("_source",       StringType(),  True),
])


# ── Ingestion functions ───────────────────────────────────────────────────────

def ingest_transactions_batch(spark: SparkSession, source_path: str,
                               bronze_path: str) -> int:
    """
    Ingest raw transaction JSON files into Bronze Delta table.
    Adds ingestion metadata (_ingestion_ts, _source) without modifying data.
    """
    logger.info(f"Ingesting transactions from {source_path}")

    df = (
        spark.read
        .option("multiline", "true")
        .schema(TRANSACTION_SCHEMA)
        .json(source_path)
        .withColumn("_ingestion_ts", F.current_timestamp())
        .withColumn("_source", F.lit(source_path))
    )

    row_count = df.count()
    logger.info(f"Ingested {row_count:,} transaction records")

    # Write to Bronze Delta table (append mode — never overwrite raw data)
    (
        df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .save(bronze_path)
    )

    logger.info(f"Bronze transactions written to {bronze_path}")
    return row_count


def ingest_customers_csv(spark: SparkSession, source_path: str,
                          bronze_path: str) -> int:
    """Ingest raw customer CSV into Bronze Delta table."""
    logger.info(f"Ingesting customers from {source_path}")

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(source_path)
        .withColumn("_ingestion_ts", F.current_timestamp())
        .withColumn("_source", F.lit(source_path))
    )

    row_count = df.count()

    (
        df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .save(bronze_path)
    )

    logger.info(f"Bronze customers written to {bronze_path}: {row_count:,} rows")
    return row_count


def validate_bronze_table(spark: SparkSession, bronze_path: str) -> dict:
    """Run basic completeness checks on Bronze Delta table."""
    df = spark.read.format("delta").load(bronze_path)

    total      = df.count()
    null_ids   = df.filter(F.col(df.columns[0]).isNull()).count()
    dup_ids    = total - df.select(df.columns[0]).distinct().count()
    oldest_ts  = df.agg(F.min("_ingestion_ts")).collect()[0][0]
    newest_ts  = df.agg(F.max("_ingestion_ts")).collect()[0][0]

    report = {
        "total_rows":       total,
        "null_primary_keys":null_ids,
        "duplicate_keys":   dup_ids,
        "oldest_ingestion": str(oldest_ts),
        "newest_ingestion": str(newest_ts),
        "quality_ok":       null_ids == 0 and dup_ids == 0,
    }
    logger.info(f"Bronze validation: {report}")
    return report
