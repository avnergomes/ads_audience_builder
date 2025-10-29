"""Tests for enrichment helpers."""

import pandas as pd

from utils.enrich import autofill_names, infer_gender


def test_autofill_names_handles_nested_objects() -> None:
    nested_first = pd.DataFrame({"value": [1]})
    nested_last = pd.DataFrame({"value": [2]})

    df = pd.DataFrame(
        {
            "fn": [nested_first],
            "ln": [nested_last],
            "email": ["alex.example@example.com"],
        }
    )

    enriched = autofill_names(df)

    assert enriched.loc[0, "fn"] == "Alex"
    assert enriched.loc[0, "ln"] == "Example"


def test_infer_gender_handles_nested_objects() -> None:
    nested_first = pd.DataFrame({"value": [1]})

    df = pd.DataFrame({"fn": [nested_first]})

    inferred = infer_gender(df)

    assert inferred.loc[0, "gender_inferred"] == "unknown"
