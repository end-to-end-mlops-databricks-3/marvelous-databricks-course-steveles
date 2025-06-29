# Databricks notebook source
# MAGIC %pip install mlops_course-0.0.1-py3-none-any.whl

# COMMAND ----------

# MAGIC %pip install -e ..

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# fix the system path.
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent / "src"))

# COMMAND ----------

# Configure tracking uri
import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from rdw.config import ProjectConfig, Tags
from rdw.models.feature_lookup_model import FeatureLookUpModel

# Configure tracking uri
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "abcd12345", "branch": "feature/fe_model"}
tags = Tags(**tags_dict)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")  # run in DB


# COMMAND ----------

# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

# Create feature table
fe_model.create_feature_table()

# COMMAND ----------

# Define house age feature function
fe_model.define_feature_function()

# COMMAND ----------

# Load data
fe_model.load_data()

# COMMAND ----------

# Perform feature engineering
fe_model.feature_engineering()

# COMMAND ----------

# Train the model
fe_model.train()

# COMMAND ----------

# register the model
fe_model.register_model()

# COMMAND ----------

# Run predictions on the last production model
# Load test set from Delta table
spark = SparkSession.builder.getOrCreate()

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

# Drop feature lookup columns and target
X_test = test_set.drop("aantal_cilinders", "aantal_deuren", "nettomaximumvermogen", config.target)


# COMMAND ----------


# from pyspark.sql.functions import col

# X_test = X_test.withColumn("LotArea", col("LotArea").cast("int")) \
#        .withColumn("OverallCond", col("OverallCond").cast("int")) \
#        .withColumn("YearBuilt", col("YearBuilt").cast("int")) \
#        .withColumn("YearRemodAdd", col("YearRemodAdd").cast("int")) \
#        .withColumn("TotalBsmtSF", col("TotalBsmtSF").cast("int"))


# COMMAND ----------

fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# Make predictions
predictions = fe_model.load_latest_model_and_predict(X_test)

# Display predictions
logger.info(predictions)
