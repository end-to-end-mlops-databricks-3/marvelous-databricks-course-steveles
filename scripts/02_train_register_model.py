"""Script for training and registering MLFlow model """
# Databricks notebook source
import argparse

import mlflow
from loguru import logger
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession

from rdw.config import ProjectConfig, Tags

# Choose one of both below.
# from rdw.models.basic_model import BasicModel
from rdw.models.custom_model import CustomModel 

from marvelous.common import is_databricks
from dotenv import load_dotenv
import os
# COMMAND ----------
# Configure tracking uri
if not is_databricks():
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")
else:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

### Arg workaround
import sys
sys.argv = [
    'ipykernel_launcher.py', 
    '--root_path', '.',
    '--env', 'dev',
    '--git_sha', 'abc123',
    '--job_run_id', 'job-001',
    '--branch', 'feature/mlflow_model'
]

# COMMAND ----------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--branch",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}./project_config.yml"
# COMMAND ----------

# config_path = f"./project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
tags_dict = {"git_sha": args.git_sha, "branch": args.branch, "job_run_id": args.job_run_id}
tags = Tags(**tags_dict)

# Initialize model
model = CustomModel(
    config=config, tags=tags, spark=spark, code_paths=[] # ["../dist/house_price-1.0.1-py3-none-any.whl"]
)
logger.info("Model initialized.")

# Load data and prepare features
model.load_data()
model.prepare_features()
logger.info("Loaded data, prepared features.")

# Train + log the model (runs everything including MLflow logging)
model.train()
model.log_model()
logger.info("Model training completed.")

model.register_model()
logger.info("Registered model")
# COMMAND ----------
