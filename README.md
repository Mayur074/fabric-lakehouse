# 🏭 Microsoft Fabric Lakehouse — Medallion Architecture Pipeline

Production-grade **Bronze → Silver → Gold Medallion Architecture** pipeline for Microsoft Fabric / Azure Synapse Analytics. Implements Delta Lake ingestion, PySpark transformations, PII handling, data quality enforcement, and Gold aggregations optimized for Power BI Direct Lake mode.

![PySpark](https://img.shields.io/badge/PySpark-3.5-orange)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-3.2-blue)
![Fabric](https://img.shields.io/badge/Microsoft_Fabric-ready-purple)
![Azure](https://img.shields.io/badge/Azure_Synapse-compatible-blue)
![CI](https://github.com/Mayur074/fabric-lakehouse/actions/workflows/ci.yml/badge.svg)

---

## Medallion Architecture

```
Source Systems
  ├── Payment Gateway (JSON events)
  ├── CRM System (CSV exports)
  └── Event Hub (streaming — batch fallback)
        │
        ▼ BRONZE LAYER (raw, immutable)
Delta Tables: bronze/transactions · bronze/customers
  ├── Append-only (never overwrite raw)
  ├── Schema enforcement with mergeSchema
  ├── Ingestion metadata (_ingestion_ts, _source)
  └── Completeness + null validation
        │
        ▼ SILVER LAYER (cleansed, validated)
Delta Tables: silver/transactions · silver/customers
  ├── Data Quality rules (not_null, accepted_values, range)
  ├── Quarantine: failed records → quarantine/
  ├── PII masking (email, first/last name)
  ├── Deduplication (latest record per PK)
  └── Standardized types + derived fields
        │
        ▼ GOLD LAYER (business-ready)
Delta Tables: gold/revenue_kpis · gold/customer_segments
  ├── Monthly KPIs with MoM growth (window functions)
  ├── RFM segmentation: CHAMPION / LOYAL / AT_RISK / ONE_TIME / CHURNED
  ├── Partitioned by year for Power BI Direct Lake
  └── Z-ordered for optimal query performance
```

## Pipeline Orchestration

The `pipelines/medallion_pipeline.json` defines the full ADF/Fabric pipeline:

```
Ingest Bronze Transactions ─┐
                             ├─► Validate Bronze ─► Transform Silver Txn ─┐
Ingest Bronze Customers ────┘                    └─► Transform Silver Cust ─┤
                                                                             ├─► Gold Revenue KPIs
                                                                             └─► Gold Customer Segments
```

## Data Quality Framework

| Check Type | Description |
|-----------|-------------|
| `completeness` | Column non-null % >= threshold |
| `uniqueness` | Distinct values % >= threshold |
| `accepted_values` | All values in allowed set |
| `range` | Numeric values within [min, max] |
| `row_count` | Table has minimum row count |

Failed records quarantined to `quarantine/` layer with failing rule name.

## Quickstart

```bash
git clone https://github.com/Mayur074/fabric-lakehouse.git
cd fabric-lakehouse
pip install -r requirements.txt

# Generate synthetic data
python scripts/generate_and_run.py

# Run tests
pytest tests/ -v
```

**For Microsoft Fabric deployment:** Upload notebooks from `notebooks/` to your Fabric Workspace and attach them to a Lakehouse. Configure `pipelines/medallion_pipeline.json` in Data Factory.

## Project Structure

```
fabric-lakehouse/
├── notebooks/
│   ├── 01_bronze_ingestion.py      # Raw ingest → Bronze Delta (append-only)
│   ├── 02_silver_transformation.py # DQ rules, PII masking, dedup → Silver Delta
│   └── 03_gold_aggregations.py     # Revenue KPIs, RFM segments → Gold Delta
├── pipelines/
│   └── medallion_pipeline.json     # ADF/Fabric pipeline definition
├── src/
│   └── data_quality.py             # Reusable DQ framework (completeness, range, etc.)
├── scripts/
│   └── generate_and_run.py         # Synthetic data generator + local runner
├── tests/
│   └── test_fabric_pipeline.py     # 31 unit tests (DQ tests import src/data_quality.py)
└── requirements.txt
```

## Author

**Mayur Sangle** — M.S. Data Science, University of Maryland
[LinkedIn](https://linkedin.com/in/mayur-sangle-04m) • [GitHub](https://github.com/Mayur074)
