"""FeatureLookUp model implementation."""

from datetime import datetime

import mlflow
import pandas as pd
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sksurv.metrics import concordance_index_censored
from xgboost import XGBClassifier

from rdw.config import ProjectConfig, Tags


class FeatureLookUpModel:
    """A class to manage FeatureLookupModel."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize the model with project configuration."""
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.lookup_features = [
            "aantal_cilinders",
            "aantal_deuren",
            "nettomaximumvermogen",
        ]  # Hardcoded - selected at random
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.rdw_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_car_age_yrs"

        # MLflow configuration
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self) -> None:
        """Create or update the rdw_features table and populate it.

        This table stores features related to RDW cars.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (kenteken STRING NOT NULL, aantal_cilinders INT, aantal_deuren INT, nettomaximumvermogen INT);
        """)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT car_pk PRIMARY KEY(kenteken);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT kenteken, aantal_cilinders, aantal_deuren, nettomaximumvermogen FROM {self.catalog_name}.{self.schema_name}.train_set"
        )
        self.spark.sql(
            f"INSERT INTO {self.feature_table_name} SELECT kenteken, aantal_cilinders, aantal_deuren, nettomaximumvermogen FROM {self.catalog_name}.{self.schema_name}.test_set"
        )
        logger.info("✅ Feature table created and populated.")

    def define_feature_function(self) -> None:
        """Define a function to calculate the car's age.

        This function subtracts the year built from the current year.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(datum_eerste_toelating BIGINT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        from datetime import datetime
        timestamp_in_seconds = datum_eerste_toelating / 1000000000.0

        registration_datetime = datetime.fromtimestamp(timestamp_in_seconds)
        reg_year = registration_datetime.year

        current_year = datetime.now().year
        age = current_year - reg_year

        if age < 0:
            return 0

        return age
        $$
        """)

        logger.info("✅ Feature function defined.")

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Drops specified columns.
        """
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "aantal_cilinders", "aantal_deuren", "nettomaximumvermogen"
        )
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        self.train_set = self.train_set.withColumn("kenteken", self.train_set["kenteken"].cast("string"))

        logger.info("✅ Data successfully loaded.")

    def feature_engineering(self) -> None:
        """Perform feature engineering by linking data with feature tables.

        Creates a training set using FeatureLookup and FeatureFunction.
        """
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=["aantal_cilinders", "aantal_deuren", "nettomaximumvermogen"],
                    lookup_key="kenteken",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="car_age_yrs",
                    input_bindings={"datum_eerste_toelating": "datum_eerste_toelating"},
                ),
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()
        current_year = datetime.now().year

        # This seems to fix previous errors
        self.test_set["car_age_yrs"] = current_year - pd.to_datetime(self.test_set["datum_eerste_toelating"]).dt.year

        self.X_train = self.training_df[self.num_features + self.cat_features + ["car_age_yrs"]]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features + ["car_age_yrs"]]
        self.y_test = self.test_set[self.target]

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Train the model and log results to MLflow.

        Uses a pipeline with preprocessing and LightGBM regressor.
        """
        logger.info("🚀 Starting training...")

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        **self.parameters,
                    ),
                ),
            ]  # Changed from LGBM
        )
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            self.pipeline.fit(self.X_train.drop(columns=["days_alive"]), self.y_train)

            # Preds for C-index computation
            y_proba = self.pipeline.predict_proba(self.X_test.drop(columns=["days_alive"]))[:, 1]
            event_indicator = self.y_test.astype(bool)
            event_time = self.X_test[
                "car_age_yrs"
            ]  # car_age_yrs or days_alive ? it would make more sense to use the newly engineered col.
            risk_scores = y_proba  # probability of death = risk

            # Preds
            y_pred = self.pipeline.predict(self.X_test.drop(columns=["days_alive"]))

            # Evaluate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            ll = log_loss(self.y_test, y_pred)  # added
            c_index = concordance_index_censored(event_indicator, event_time, risk_scores)[0]

            logger.info(f"📊 Mean Squared Error: {mse}")
            logger.info(f"📊 Mean Absolute Error: {mae}")
            logger.info(f"📊 R2 Score: {r2}")
            logger.info(f"📊 Log loss: {ll}")
            logger.info(f"📊 Harrell's C-Statistic: {c_index}")

            # Log parameters and metrics
            mlflow.log_param("model_type", "XGBoost with preprocessing")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("log_loss", ll)
            mlflow.log_metric("harrell_c_stat", c_index)
            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=self.pipeline,
                flavor=mlflow.sklearn,
                artifact_path="xgb-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )

    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/xgb-pipeline-model-fe",
            name=f"{self.catalog_name}.{self.schema_name}.rdw_model_fe",
            tags=self.tags,
        )

        # Fetch the latest version dynamically
        latest_version = registered_model.version

        client = MlflowClient()
        client.set_registered_model_alias(
            name=f"{self.catalog_name}.{self.schema_name}.rdw_model_fe",
            alias="latest-model",
            version=latest_version,
        )

        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.rdw_model_fe@latest-model"

        predictions = self.fe.score_batch(model_uri=model_uri, df=X)
        return predictions
