"""Data preprocessing module."""

import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from rdw.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations.

    This class handles data preprocessing, splitting, and saving to Databricks tables.
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the DataFrame stored in self.df.

        This method handles missing values, converts data types, and performs feature engineering.
        """
        self.df["datum_eerste_toelating"] = pd.to_datetime(self.df["datum_eerste_toelating"], errors="coerce")
        self.df["vervaldatum_apk"] = pd.to_datetime(self.df["vervaldatum_apk"], errors="coerce")

        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Fill missing numeric values with median
        for col in num_features:
            median_val = self.df[col].median()
            self.df[col] = self.df[col].fillna(median_val)

        # Fill categorical NaNs with placeholder
        cat_features = self.config.cat_features
        for col in cat_features:
            self.df[col] = self.df[col].fillna("Unknown")
            self.df[col] = self.df[col].astype("category")

        # Ensure target column is numeric
        self.df["is_dead"] = pd.to_numeric(self.df["is_dead"], errors="coerce").fillna(0).astype(int)

        # Extract target and relevant features
        target = self.config.target
        relevant_columns = cat_features + num_features + [target]
        self.df = self.df[relevant_columns]

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


def generate_synthetic_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 500) -> pd.DataFrame:
    """Generate synthetic vehicle dataset based on statistical distribution of input DataFrame.

    Supports optional data drift injection for experimentation or model testing.

    :param df: Source vehicle DataFrame
    :param drift: Inject artificial drift into select features
    :param num_rows: Number of synthetic records to generate
    :return: Synthetic DataFrame
    """
    synthetic_data = pd.DataFrame()

    for column in df.columns:
        if column == "kenteken":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            synthetic_data[column] = np.random.normal(df[column].mean(), df[column].std(), num_rows)

            # Clamp values to be non-negative where needed
            if column in {
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
            }:
                synthetic_data[column] = np.maximum(0, synthetic_data[column])

        elif pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(
                df[column].dropna().unique(), num_rows, p=df[column].value_counts(normalize=True)
            )

        elif pd.api.types.is_datetime64_any_dtype(df[column]) or "datum" in column.lower():
            try:
                parsed_dates = pd.to_datetime(df[column], errors="coerce").dropna()
                if not parsed_dates.empty:
                    min_date = parsed_dates.min()
                    max_date = parsed_dates.max()
                    synthetic_data[column] = pd.to_datetime(np.random.randint(min_date.value, max_date.value, num_rows))
                else:
                    synthetic_data[column] = pd.NaT
            except Exception:
                synthetic_data[column] = pd.NaT

        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)

    # Cast back to appropriate int types for specific columns if needed
    int_columns = ["massa_ledig_voertuig", "aantal_wielen", "days_alive", "is_dead"]
    for col in int_columns:
        if col in synthetic_data.columns:
            synthetic_data[col] = synthetic_data[col].astype(int)

    timestamp_base = int(time.time() * 1000)
    synthetic_data["synthetic_id"] = [str(timestamp_base + i) for i in range(num_rows)]

    # Drift injection
    if drift:
        drift_features = ["massa_rijklaar", "vermogen_massarijklaar"]
        for feature in drift_features:
            if feature in synthetic_data.columns:
                synthetic_data[feature] *= 1.5  # inflate

        if "vervaldatum_apk" in synthetic_data.columns:
            future_dates = pd.date_range(start=pd.Timestamp.now(), periods=num_rows, freq="D")
            synthetic_data["vervaldatum_apk"] = np.random.choice(future_dates, num_rows)

    return synthetic_data


def generate_test_data(df: pd.DataFrame, drift: bool = False, num_rows: int = 100) -> pd.DataFrame:
    """Generate test data matching input DataFrame distributions with optional drift."""
    return generate_synthetic_data(df, drift, num_rows)
