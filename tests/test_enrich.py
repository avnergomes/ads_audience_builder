"""Tests for enrichment helpers."""

import pandas as pd

from utils.enrich import autofill_names, infer_gender


def test_autofill_names_respects_existing_columns() -> None:
    df = pd.DataFrame(
        {
            "fn": ["Existing"],
            "ln": ["Name"],
            "email": ["alex.example@example.com"],
        }
    )

    enriched = autofill_names(df)

    assert enriched.loc[0, "fn"] == "Existing"
    assert enriched.loc[0, "ln"] == "Name"


def test_autofill_names_uses_email_only_when_missing_columns() -> None:
    df = pd.DataFrame({"email": ["alex.example@example.com"]})

    enriched = autofill_names(df)

    assert enriched.loc[0, "fn"] == "Alex"
    assert enriched.loc[0, "ln"] == "Example"


def test_autofill_names_uses_nameapi_when_available(monkeypatch) -> None:
    df = pd.DataFrame({"email": ["something@example.com"]})

    monkeypatch.setattr(
        "utils.enrich._nameapi_lookup", lambda email: ("Api", "Result")
    )

    enriched = autofill_names(df)

    assert enriched.loc[0, "fn"] == "Api"
    assert enriched.loc[0, "ln"] == "Result"


def test_infer_gender_handles_nested_objects() -> None:
    nested_first = pd.DataFrame({"value": [1]})

    df = pd.DataFrame({"fn": [nested_first]})

    inferred = infer_gender(df)

    assert inferred.loc[0, "gender_inferred"] == "unknown"
