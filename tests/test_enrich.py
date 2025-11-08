"""Tests for enrichment helpers."""

from typing import Optional

import pandas as pd

from utils.enrich import (
    _extract_nameapi_names,
    _nameapi_lookup,
    _nameapi_lookup_single,
    autofill_names,
    derive_name_from_email,
    infer_gender,
)


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


def test_extract_nameapi_names_strips_digits() -> None:
    first, last = _extract_nameapi_names(
        {"result": {"given_name": "Caroline123", "surname": "Smith456"}}
    )

    assert first == "Caroline"
    assert last == "Smith"


def test_nameapi_lookup_checks_variants(monkeypatch) -> None:
    email = "tkaur0820@example.com"
    calls: list[str] = []

    def fake_lookup_single(candidate: str) -> tuple[Optional[str], Optional[str]]:
        calls.append(candidate)
        if candidate == "tkaur@example.com":
            return "Tej", "Kaur"
        return None, None

    _nameapi_lookup.cache_clear()
    _nameapi_lookup_single.cache_clear()
    monkeypatch.setattr("utils.enrich._nameapi_lookup_single", fake_lookup_single)

    first, last = derive_name_from_email(email)

    assert first == "Tej"
    assert last == "Kaur"
    assert calls[0] == email
    assert "tkaur@example.com" in calls
