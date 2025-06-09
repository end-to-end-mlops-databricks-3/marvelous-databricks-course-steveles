"""Tests for mlflow serve."""

import json

import pandas as pd
import pytest
import requests
from conftest import CATALOG_DIR
from loguru import logger

BASE_URL = "http://127.0.0.1:5088"

test_data = {
    "kenteken": "R662GG",
    "voertuigsoort": "Personenauto",
    "merk": "VOLKSWAGEN",
    "handelsbenaming": "GOLF",
    "inrichting": "stationwagen",
    "eerste_kleur": "BLAUW",
    "type": "CD",
    "tellerstandoordeel": "Geen oordeel",
    "brandstof_omschrijving": "Benzine",
    "aantal_zitplaatsen": 5.0,
    "aantal_cilinders": 4.0,
    "cilinderinhoud": 1984.0,
    "massa_ledig_voertuig": 1451,
    "massa_rijklaar": 1551.0,
    "catalogusprijs": 62436.0,
    "aantal_deuren": 5.0,
    "aantal_wielen": 4,
    "lengte": 429.0,
    "breedte": 179.0,
    "vermogen_massarijklaar": 0.15,
    "wielbasis": 263.0,
    "days_alive": 1461,
    "nettomaximumvermogen": 235.0,
    "vervaldatum_apk": 1740355200000000000,
    "datum_eerste_toelating": 1614124800000000000,
    "is_dead": 1
}

pandas_df = pd.DataFrame([test_data])

payload_dataframe_split = json.dumps({"dataframe_split": pandas_df.to_dict(orient="split")})
payload_dataframe_records = json.dumps({"dataframe_records": pandas_df.to_dict(orient="records")})


@pytest.mark.ci_exclude
def test_inference_server_health() -> None:
    """Test that the inference server health endpoint is reachable.

    Sends a GET request to the health endpoint and asserts a 200 status code is returned.
    """
    response = requests.get(f"{BASE_URL}/health")
    logger.info(f"Received {response.status_code}.")
    assert response.status_code == 200


@pytest.mark.ci_exclude
def test_inference_server_ping() -> None:
    """Test that the inference server ping endpoint is reachable.

    Sends a GET request to the ping endpoint and asserts a 200 status code is returned.
    """
    response = requests.get(f"{BASE_URL}/ping")
    logger.info(f"Received {response.status_code}.")
    assert response.status_code == 200


@pytest.mark.ci_exclude
def test_inference_server_version() -> None:
    """Test that the inference server version endpoint returns the expected version.

    Sends a GET request to the version endpoint, asserts a 200 status code, and checks the version string.
    """
    response = requests.get(f"{BASE_URL}/version")
    logger.info(f"Received {response.status_code} with response of '{response.text}'.")
    assert response.status_code == 200
    assert response.text == "2.17.0"


@pytest.mark.ci_exclude
def test_inference_server_invocations_with_dataframe_split() -> None:
    """Test that the inference server correctly handles DataFrame split payloads.

    Sends a POST request with a DataFrame split payload and verifies the response contains a list of float predictions.
    """
    response = requests.post(
        f"{BASE_URL}/invocations", data=payload_dataframe_split, headers={"Content-Type": "application/json"}, timeout=2
    )
    logger.info(f"Received {response.status_code} with response of '{response.text}'.")
    assert response.status_code == 200
    logger.info(f"Received {response.json()}")
    values = response.json()["predictions"]
    assert isinstance(values, list)
    assert isinstance(values[0], float)


@pytest.mark.ci_exclude
def test_inference_server_invocations_with_dataframe_records() -> None:
    """Test that the inference server correctly handles DataFrame records payloads.

    Sends a POST request with a DataFrame records payload and verifies the response contains a list of float predictions.
    """
    response = requests.post(
        f"{BASE_URL}/invocations",
        data=payload_dataframe_records,
        headers={"Content-Type": "application/json"},
        timeout=2,
    )
    logger.info(f"Received {response.status_code} with response of '{response.text}'.")
    assert response.status_code == 200
    logger.info(f"Received {response.json()}")
    values = response.json()["predictions"]
    assert isinstance(values, list)
    assert isinstance(values[0], float)


@pytest.mark.ci_exclude
def test_inference_server_invocations_with_dataframe_records_should_fail_when_contact_request_violation() -> None:
    """Test that inference server invocations with incomplete DataFrame records fail as expected.

    Drops each column from the DataFrame in turn and verifies that the server returns a 400 error.
    """
    for col in pandas_df.columns.to_list():
        tmp_df = pandas_df.drop(columns=[col])

        tmp_payload_dataframe_records = json.dumps({"dataframe_records": tmp_df.to_dict(orient="records")})
        logger.info(f"Testing with {col} dropped.")
        response = requests.post(
            f"{BASE_URL}/invocations",
            data=tmp_payload_dataframe_records,
            headers={"Content-Type": "application/json"},
            timeout=2,
        )
        logger.info(f"Received {response.status_code} with response of '{response.text}'.")
        assert response.status_code == 400


@pytest.mark.ci_exclude
def test_infererence_server_invocations_with_full_dataframe() -> None:
    """Test that the inference server processes a full DataFrame and returns correct predictions.

    Loads test data, sends a POST request, and verifies the response contains a list of float predictions of correct length.
    """
    test_set = pd.read_csv(f"{CATALOG_DIR.as_posix()}/test_set.csv")
    input_data = test_set.drop(columns=["is_dead"])
    input_data = input_data.where(input_data.notna(), None)  # noqa
    input_data = input_data.to_dict(orient="records")
    payload = json.dumps({"dataframe_records": input_data})

    response = requests.post(
        f"{BASE_URL}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
        timeout=2,
    )
    logger.info(f"Received {response.status_code} with response of '{response.text}'.")
    assert response.status_code == 200

    values = response.json()["predictions"]
    assert isinstance(values, list)
    assert len(values) == len(input_data)
    assert all(isinstance(value, float) for value in values)