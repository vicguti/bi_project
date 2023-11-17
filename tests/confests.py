import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

@pytest.fixture(scope="module")
def temp_directory(tmpdir_factory):
    dir = tmpdir_factory
    return dir

@pytest.fixture(scope="module")
def spark(temp_directory) -> SparkSession:
    builder = (
        SparkSession.builder.appName("TestSparkSession")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "5")
        .config("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .config("spark.sql.sources.default", "delta")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.warehouse.dir", str(temp_directory.mktemp("warehouse_dir")))
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    return spark