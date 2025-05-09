from src.rdw.download_dataset import chunked_download, convert_illegal_column_characters
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

volume_path = f"/Volumes/mlops_dev/steven/rdw/data/gekentekende_voertuigen.csv"
delta_path = f"mlops_dev.steven.rdw.data.gekentekende_voertuigen"


chunked_download(url="https://opendata.rdw.nl/api/views/m9d7-ebf2/rows.csv?accessType=DOWNLOAD", output_path=volume_path)

df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(volume_path
)

df = convert_illegal_column_characters(df)

df.write.format("delta") \
    .mode("overwrite") \
    .option("mergeSchema", "false") \
    .option("overwriteSchema", "false") \
    .saveAsTable(delta_path)