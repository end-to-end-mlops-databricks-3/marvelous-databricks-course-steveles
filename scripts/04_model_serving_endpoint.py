# Databricks notebook source
# MAGIC %pip install house_price-1.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------


# COMMAND ----------

import os
import time

import requests
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from rdw.config import ProjectConfig
from rdw.serving.model_serving import ModelServing

# spark session

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

# get environment variables
os.environ["DBR_TOKEN"] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["DBR_HOST"] = spark.conf.get("spark.databricks.workspaceUrl")

# Load project config
config = ProjectConfig.from_yaml(config_path="./project_config.yml")  # .. in DB
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Initialize feature store manager
model_serving = ModelServing(
    model_name=f"{catalog_name}.{schema_name}.rdw_model_basic", endpoint_name="rdw-model-serving"
)

# COMMAND ----------

# Deploy the model serving endpoint
model_serving.deploy_or_update_serving_endpoint()


# COMMAND ----------

# Create a sample request body
required_columns = [
    "kenteken",
    "voertuigsoort",
    "merk",
    "handelsbenaming",
    "inrichting",
    "eerste_kleur",
    "type",
    "tellerstandoordeel",
    "brandstof_omschrijving",
    "aantal_zitplaatsen",
    "aantal_cilinders",
    "cilinderinhoud",
    "massa_ledig_voertuig",
    "massa_rijklaar",
    "catalogusprijs",
    "aantal_deuren",
    "aantal_wielen",
    "lengte",
    "breedte",
    "vermogen_massarijklaar",
    "wielbasis",
    "days_alive",
    "nettomaximumvermogen",
    "vervaldatum_apk",
    "datum_eerste_toelating",
]


# Sample 1000 records from the training set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").toPandas()

# Sample 100 records from the training set
sampled_records = test_set[required_columns].sample(n=100, replace=True).to_dict(orient="records")
dataframe_records = [[record] for record in sampled_records]

# COMMAND ----------

# Call the endpoint with one sample record

# """
# Each dataframe record in the request body should be list of json with columns looking like:

# [{'LotFrontage': 78.0,
#   'LotArea': 9317,
#   'OverallQual': 6,
#   'OverallCond': 5,
#   'YearBuilt': 2006,
#   'Exterior1st': 'VinylSd',
#   'Exterior2nd': 'VinylSd',
#   'MasVnrType': 'None',
#   'Foundation': 'PConc',
#   'Heating': 'GasA',
#   'CentralAir': 'Y',
#   'SaleType': 'WD',
#   'SaleCondition': 'Normal'}]
# """


def call_endpoint(record: any) -> tuple[int, str]:
    """Call the model serving endpoint with a given input record."""
    serving_endpoint = f"https://{os.environ['DBR_HOST']}/serving-endpoints/rdw-model-serving/invocations"

    response = requests.post(
        serving_endpoint,
        headers={"Authorization": f"Bearer {os.environ['DBR_TOKEN']}"},
        json={"dataframe_records": record},
    )
    return response.status_code, response.text


status_code, response_text = call_endpoint(dataframe_records[0])
print(f"Response Status: {status_code}")
print(f"Response Text: {response_text}")

# COMMAND ----------

# Load test
for i in range(len(dataframe_records)):
    status_code, response_text = call_endpoint(dataframe_records[i])
    print(f"Response Status: {status_code}")
    print(f"Response Text: {response_text}")
    time.sleep(0.2)
