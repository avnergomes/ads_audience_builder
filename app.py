"""Streamlit application for the Smart Custom Audience Builder."""

from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd
import streamlit as st

from utils import cleaning, enrich, export_google, export_meta, export_tiktok, segmentation, ingest


st.set_page_config(page_title="Smart Custom Audience Builder", layout="wide")

st.title("🧠 Smart Custom Audience Builder")
st.write("Upload, clean, enrich, segment, and export your audience lists for Meta, Google, and TikTok Ads.")


def load_uploaded_file(file) -> pd.DataFrame:
    """Load an uploaded file into a DataFrame with robust CSV handling."""

    filename = getattr(file, "name", "") or ""
    suffix = filename.lower()

    if suffix.endswith(".xlsx"):
        if hasattr(file, "seek"):
            file.seek(0)
        return pd.read_excel(file, dtype=str)

    return ingest.read_audience_csv(file)


def _normalise_header(value: object) -> str:
    """Return a lower-cased string representation for fuzzy header matching."""

    if value is None:
        return ""
    return str(value).strip().lower()


EXPECTED_FIELDS: Dict[str, List[str]] = {
    "email": ["email", "e-mail", "mail"],
    "phone": ["phone", "phone number", "mobile"],
    "fn": ["first name", "fn"],
    "ln": ["last name", "ln"],
    "full_name": ["name", "full name", "fullname"],
    "ct": ["city", "ct"],
    "st": ["state", "region", "st"],
    "zip": ["zip", "postal", "zip code"],
    "country": ["country", "country code"],
    "meta_lead_id": ["lead_id", "lead id", "uid"],
    "source": ["source", "utm_source", "campaign"],
}


def field_mapping_ui(df: pd.DataFrame) -> Dict[str, str]:
    st.subheader("🔁 Field Mapping")
    st.write("Match the columns from your file to the expected destination fields.")

    mapping: Dict[str, str] = {}
    columns = [column for column in df.columns]

    options = ["Not mapped"] + columns
    mapping_columns = list(EXPECTED_FIELDS.keys())
    for chunk_start in range(0, len(mapping_columns), 3):
        row_columns = st.columns(3)
        for idx, field in enumerate(mapping_columns[chunk_start : chunk_start + 3]):
            with row_columns[idx]:
                hints = EXPECTED_FIELDS[field]
                default_option = "Not mapped"
                normalised_hints = {_normalise_header(hint) for hint in hints}
                for column in columns:
                    if _normalise_header(column) in normalised_hints:
                        default_option = column
                        break
                selection = st.selectbox(
                    f"{field}",
                    options,
                    index=options.index(default_option) if default_option in options else 0,
                    help=f"Expected examples: {', '.join(hints)}",
                )
                if selection != "Not mapped":
                    mapping[selection] = field
    return mapping


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    if not mapping:
        return df.copy()
    return df.rename(columns=mapping)


def slugify(value: object) -> str:
    text = value if isinstance(value, str) else f"{value}"
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "audience"


uploaded_file = st.file_uploader(
    "📤 Drag & drop or browse a customer list (CSV/XLSX)", type=["csv", "xlsx"]
)

if uploaded_file:
    raw_df = load_uploaded_file(uploaded_file)
    st.success(f"Loaded {len(raw_df):,} rows from {uploaded_file.name}.")

    st.subheader("👀 Raw Preview")
    st.dataframe(raw_df.head())

    mapping = field_mapping_ui(raw_df)
    mapped_df = apply_mapping(raw_df, mapping)

    cleaned_df, stats, summary = cleaning.clean_dataframe(mapped_df)

    st.subheader("🧼 Cleaning Summary")
    cols = st.columns(5)
    cols[0].metric("Rows imported", stats["initial_rows"])
    cols[1].metric("Invalid emails removed", stats["invalid_emails"])
    cols[2].metric("Email corrections applied", stats.get("email_corrections", 0))
    cols[3].metric("Missing emails", stats.get("missing_emails", 0))
    cols[4].metric("Invalid phones found", stats.get("invalid_phones", 0))

    cols = st.columns(3)
    cols[0].metric("Email parsing issues fixed", stats.get("email_parsing_errors", 0))
    cols[1].metric("Duplicates removed", stats["duplicates_removed"])
    cols[2].metric("Rows without contact removed", stats.get("rows_without_contact", 0))

    st.metric("Rows after cleaning", stats["final_rows"])

    if summary:
        st.write("Field completeness snapshot")
        st.table(pd.DataFrame(summary))

    st.subheader("✨ Enrichment")
    col1, col2, col3 = st.columns(3)
    autofill = col1.checkbox("Autofill names from email", value=True)
    infer_gender = col2.checkbox("Infer gender from first name")
    zip_enrichment = col3.checkbox("Enrich City/State from ZIP")
    overwrite_zip = col3.checkbox("Overwrite existing City/State", value=False, key="overwrite_zip")

    if autofill:
        cleaned_df = enrich.autofill_names(cleaned_df)
    if infer_gender:
        cleaned_df = enrich.infer_gender(cleaned_df)
    if zip_enrichment:
        cleaned_df = enrich.enrich_from_zip(cleaned_df, overwrite_existing=overwrite_zip)

    st.subheader("🏷️ Audience tagging & segmentation")
    source_tag = st.text_input("Audience source tag", help="e.g. Instagram, Signup Form, Meta Leadgen")
    if source_tag:
        cleaned_df = segmentation.tag_source(cleaned_df, source_tag)

    enable_ab = st.checkbox("Create 50/50 A/B test splits", value=False)
    ab_fraction = st.slider(
        "Split A size", min_value=0.1, max_value=0.9, value=0.5, step=0.05, disabled=not enable_ab
    )

    segments = segmentation.available_segments(cleaned_df, ab_fraction, enable_ab)
    segments = segmentation.tag_segments(segments)

    st.subheader("📊 Preview cleaned data")
    tabs = st.tabs(list(segments.keys()))
    for tab, (segment_name, segment_df) in zip(tabs, segments.items()):
        with tab:
            st.write(f"Rows: {len(segment_df):,}")
            st.dataframe(segment_df.head(20))

    st.subheader("📦 Export ready files")
    hash_meta = st.checkbox("Hash personally identifiable fields for Meta", value=True)

    for segment_name, segment_df in segments.items():
        st.write(f"### {segment_name}")

        meta_df = export_meta.export_meta(segment_df, hash_fields=hash_meta)
        google_df = export_google.export_google(segment_df)
        tiktok_df = export_tiktok.export_tiktok(segment_df)

        col_meta, col_google, col_tiktok = st.columns(3)

        with col_meta:
            st.download_button(
                label="⬇️ Meta CSV",
                data=meta_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{slugify(segment_name)}_meta.csv",
                mime="text/csv",
            )

        with col_google:
            st.download_button(
                label="⬇️ Google CSV",
                data=google_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{slugify(segment_name)}_google.csv",
                mime="text/csv",
            )

        with col_tiktok:
            st.download_button(
                label="⬇️ TikTok CSV",
                data=tiktok_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{slugify(segment_name)}_tiktok.csv",
                mime="text/csv",
            )

    st.info("Exports include Meta Lead IDs, audience source tags, and are ready for upload to each platform.")
