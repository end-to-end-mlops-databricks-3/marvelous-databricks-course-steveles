import requests
import re
from pyspark.sql import DataFrame

def chunked_download(url: str, output_path: str, chunk_size: int = 512 * 1024) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # e.g., /Volumes/main_catalog/rdw/raw
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
            
def to_snake_case(col_name: str) -> str:
    c = col_name.lower()
    c = re.sub(r'[^a-z0-9]+', '_', c)
    c = re.sub(r'_+', '_', c)
    c = c.strip('_')
    return c

def convert_illegal_column_characters(df: DataFrame):
    for col in df.columns:
        df = df.withColumnRenamed(col, to_snake_case(col))
    return df