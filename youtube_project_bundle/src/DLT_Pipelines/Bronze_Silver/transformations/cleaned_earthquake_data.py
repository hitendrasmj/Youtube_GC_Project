from pyspark.sql.functions import ( col, from_json, explode, current_timestamp, from_unixtime )
import dlt
from pyspark.sql.types import *

# This file defines a sample transformation.
# Edit the sample below or add new transformations
# using "+ Add" in the file browser.

catalog_name = spark.conf.get("catalog_name")
volume_path = f"/Volumes/{catalog_name}/bronze/earthquake_data"
primary_key = "id"

properties_schema = StructType(
    [
        StructField("mag", StringType(), True),
        StructField("place", StringType(), True),
        StructField("time", StringType(), True),
        StructField("updated", StringType(), True),
        StructField("tz", StringType(), True),
        StructField("url", StringType(), True),
        StructField("detail", StringType(), True),
        StructField("felt", StringType(), True),
        StructField("cdi", StringType(), True),
        StructField("mmi", StringType(), True),
        StructField("alert", StringType(), True),
        StructField("status", StringType(), True),
        StructField("tsunami", StringType(), True),
        StructField("sig", StringType(), True),
        StructField("net", StringType(), True),
        StructField("code", StringType(), True),
        StructField("ids", StringType(), True),
        StructField("sources", StringType(), True),
        StructField("types", StringType(), True),
        StructField("nst", StringType(), True),
        StructField("dmin", StringType(), True),
        StructField("rms", StringType(), True),
        StructField("gap", StringType(), True),
        StructField("magType", StringType(), True),
        StructField("type", StringType(), True),
        StructField("title", StringType(), True),
    ]
)

geometry_schema = StructType(
    [
        StructField("coordinates", ArrayType(DoubleType()), True),
    ]
)

feature_schema = StructType(
    [
        StructField("id", StringType(), True),
        StructField("properties", StructType(properties_schema), True),
        StructField("geometry", StructType(geometry_schema), True),
    ]
)

schema = ArrayType(feature_schema)


@dlt.view(name="earthquake_data_vw")
def earthquake_data():
    df = (
        spark.readStream.format("cloudfiles")
        .option("cloudFiles.format", "json")
        .load(volume_path)
        .withColumn("_load_timestamp", current_timestamp())
    )
    df = df.withColumn("parsed_data", from_json(col("features"), schema))
    df = df.select(explode(col("parsed_data")).alias("features"),"_load_timestamp")
    df = df.select(
        "features.properties.*",
        "features.id",
        col("features.geometry.coordinates")[0].alias("lognitude"),
        col("features.geometry.coordinates")[1].alias("latitude"),
        col("features.geometry.coordinates")[2].alias("depth"),
        "_load_timestamp",
    )
    df = (
        df.withColumn("time", from_unixtime(col("time") / 1000).cast("timestamp"))
        .withColumn("updated", from_unixtime(col("updated") / 1000).cast("timestamp"))
        .withColumn("mag", col("mag").cast("double"))
        .withColumn("nst", col("nst").cast("double"))
        .withColumn("tsunami", col("tsunami").cast("double"))
        .withColumn("felt", col("felt").cast("double"))
    )
    return df

dlt.create_streaming_table(name="earthquake_data_final")
dlt.apply_changes(
    target="earthquake_data_final",
    source="earthquake_data_vw",
    keys=[primary_key],
    sequence_by="_load_timestamp",
    stored_as_scd_type="1"
)
