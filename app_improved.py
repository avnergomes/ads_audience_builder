"""Streamlit application for the Smart Custom Audience Builder - Improved Version."""

from __future__ import annotations

import re
from typing import Dict, List

import pandas as pd
import streamlit as st

from utils import cleaning, enrich, export_google, export_meta, export_tiktok, segmentation, ingest


# Page configuration
st.set_page_config(
    page_title="Smart Custom Audience Builder",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 Smart Custom Audience Builder</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload, clean, enrich, segment, and export your audience lists for Meta, Google, and TikTok Ads.</div>', unsafe_allow_html=True)


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
    "email": ["email", "e-mail", "mail", "email address"],
    "phone": ["phone", "phone number", "mobile", "cell"],
    "fn": ["first name", "fn", "firstname", "first"],
    "ln": ["last name", "ln", "lastname", "last", "surname"],
    "full_name": ["name", "full name", "fullname", "full_name"],
    "ct": ["city", "ct", "town"],
    "st": ["state", "region", "st", "province"],
    "zip": ["zip", "postal", "zip code", "zipcode", "postal code"],
    "country": ["country", "country code"],
    "meta_lead_id": ["lead_id", "lead id", "uid", "id"],
    "source": ["source", "utm_source", "campaign"],
}


def field_mapping_ui(df: pd.DataFrame) -> Dict[str, str]:
    """Display field mapping interface."""
    st.markdown("### 🔁 Field Mapping")
    st.markdown("Match the columns from your file to the expected destination fields.")
    
    with st.expander("ℹ️ Field Mapping Help", expanded=False):
        st.markdown("""
        **Required fields for best results:**
        - **Email**: Primary identifier for most platforms
        - **Phone**: Secondary identifier (E.164 format preferred)
        - **First Name (fn)** & **Last Name (ln)**: Improves match rates
        
        **Optional but recommended:**
        - **City**, **State**, **ZIP**: Location targeting
        - **Country**: International campaigns
        - **Source**: Track audience origin
        """)

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
                
                # Show icon for required fields
                label = f"{field} {'⭐' if field in ['email', 'phone'] else ''}"
                selection = st.selectbox(
                    label,
                    options,
                    index=options.index(default_option) if default_option in options else 0,
                    help=f"Common names: {', '.join(hints[:3])}",
                    key=f"mapping_{field}"
                )
                if selection != "Not mapped":
                    mapping[selection] = field
    
    return mapping


def apply_mapping(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Apply column mapping to DataFrame."""
    if not mapping:
        return df.copy()
    return df.rename(columns=mapping)


def slugify(value: object) -> str:
    """Convert text to URL-safe slug."""
    text = value if isinstance(value, str) else f"{value}"
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value or "audience"


def display_cleaning_stats(stats: Dict, summary: List[Dict]):
    """Display cleaning statistics in an organized way."""
    st.markdown("### 🧼 Cleaning Summary")
    
    # Main metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows Imported", f"{stats['initial_rows']:,}")
    with col2:
        st.metric("Rows After Cleaning", f"{stats['final_rows']:,}", 
                 delta=f"{stats['final_rows'] - stats['initial_rows']:,}")
    with col3:
        retention_rate = (stats['final_rows'] / stats['initial_rows'] * 100) if stats['initial_rows'] > 0 else 0
        st.metric("Retention Rate", f"{retention_rate:.1f}%")
    with col4:
        st.metric("Duplicates Removed", f"{stats['duplicates_removed']:,}")
    
    # Detailed statistics in expander
    with st.expander("📊 Detailed Cleaning Statistics", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Email Processing:**")
            st.write(f"- Invalid emails removed: {stats['invalid_emails']:,}")
            st.write(f"- Email corrections applied: {stats.get('email_corrections', 0):,}")
            st.write(f"- Missing emails: {stats.get('missing_emails', 0):,}")
            st.write(f"- Parsing issues fixed: {stats.get('email_parsing_errors', 0):,}")
        
        with col2:
            st.markdown("**Other Cleaning:**")
            st.write(f"- Invalid phones found: {stats.get('invalid_phones', 0):,}")
            st.write(f"- Rows without contact info: {stats.get('rows_without_contact', 0):,}")
    
    # Field completeness
    if summary:
        st.markdown("**Field Completeness:**")
        summary_df = pd.DataFrame(summary)
        summary_df['completeness'] = (summary_df['populated'] / (summary_df['populated'] + summary_df['missing']) * 100).round(1)
        
        col1, col2, col3 = st.columns(3)
        for idx, row in summary_df.iterrows():
            with [col1, col2, col3][idx % 3]:
                st.metric(
                    row['column'].upper(),
                    f"{row['populated']:,} populated",
                    f"{row['completeness']:.1f}%"
                )


def enrichment_section(df: pd.DataFrame) -> pd.DataFrame:
    """Display enrichment options and process data."""
    st.markdown("### ✨ Data Enrichment")
    
    st.markdown("""
    Enrich your audience data to improve match rates and targeting precision.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Name Enrichment:**")
        autofill = st.checkbox(
            "🔤 Autofill names from email",
            value=True,
            help="Extract first and last names from email addresses when missing"
        )
        infer_gender = st.checkbox(
            "⚧ Infer gender from first name",
            value=False,
            help="Use name database and API to predict gender (may have API limits)"
        )
    
    with col2:
        st.markdown("**Location Enrichment:**")
        zip_enrichment = st.checkbox(
            "📍 Enrich City/State from ZIP",
            value=False,
            help="Lookup city and state information from ZIP codes"
        )
        overwrite_zip = st.checkbox(
            "♻️ Overwrite existing City/State",
            value=False,
            disabled=not zip_enrichment,
            help="Replace existing location data with ZIP lookup results"
        )
    
    # Process enrichments with progress indicators
    enriched_df = df.copy()
    total_steps = sum([autofill, infer_gender, zip_enrichment])
    
    if total_steps > 0:
        progress_bar = st.progress(0)
        status_text = st.empty()
        current_step = 0
        
        if autofill:
            current_step += 1
            status_text.text(f"Step {current_step}/{total_steps}: Extracting names from emails...")
            progress_bar.progress(current_step / total_steps)
            enriched_df = enrich.autofill_names(enriched_df)
        
        if infer_gender:
            current_step += 1
            status_text.text(f"Step {current_step}/{total_steps}: Inferring gender from names...")
            progress_bar.progress(current_step / total_steps)
            with st.spinner("Calling gender API (this may take a moment)..."):
                enriched_df = enrich.infer_gender(enriched_df)
        
        if zip_enrichment:
            current_step += 1
            status_text.text(f"Step {current_step}/{total_steps}: Enriching location from ZIP codes...")
            progress_bar.progress(current_step / total_steps)
            enriched_df = enrich.enrich_from_zip(enriched_df, overwrite_existing=overwrite_zip)
        
        status_text.text("✅ Enrichment complete!")
        progress_bar.progress(1.0)
        
        # Show enrichment results
        if 'fn' in enriched_df.columns or 'ln' in enriched_df.columns:
            names_added = 0
            if 'fn' in enriched_df.columns:
                names_added += enriched_df['fn'].notna().sum() - df.get('fn', pd.Series()).notna().sum()
            if 'ln' in enriched_df.columns:
                names_added += enriched_df['ln'].notna().sum() - df.get('ln', pd.Series()).notna().sum()
            if names_added > 0:
                st.success(f"✅ Added {names_added:,} name fields")
        
        if 'gender_inferred' in enriched_df.columns:
            genders_added = enriched_df['gender_inferred'].notna().sum()
            if genders_added > 0:
                st.success(f"✅ Inferred gender for {genders_added:,} records")
    
    return enriched_df


def segmentation_section(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Display segmentation options and create segments."""
    st.markdown("### 🏷️ Audience Tagging & Segmentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_tag = st.text_input(
            "📌 Audience source tag",
            placeholder="e.g., Instagram, Signup Form, Meta Leadgen",
            help="Tag to identify the source of this audience"
        )
    
    with col2:
        enable_ab = st.checkbox(
            "🔀 Create A/B test splits",
            value=False,
            help="Split audience into two equal groups for testing"
        )
        
        if enable_ab:
            ab_fraction = st.slider(
                "Split A size (%)",
                min_value=10,
                max_value=90,
                value=50,
                step=5,
                help="Percentage of audience in Split A"
            ) / 100
        else:
            ab_fraction = 0.5
    
    # Apply source tag
    if source_tag:
        df = segmentation.tag_source(df, source_tag)
    
    # Create segments
    segments = segmentation.available_segments(df, ab_fraction, enable_ab)
    segments = segmentation.tag_segments(segments)
    
    # Display segment info
    st.info(f"📦 Created {len(segments)} segment(s): {', '.join(segments.keys())}")
    
    return segments


def preview_section(segments: Dict[str, pd.DataFrame]):
    """Display data preview for each segment."""
    st.markdown("### 📊 Data Preview")
    
    tabs = st.tabs(list(segments.keys()))
    for tab, (segment_name, segment_df) in zip(tabs, segments.items()):
        with tab:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.metric("Total Rows", f"{len(segment_df):,}")
            with col2:
                # Show key stats
                has_email = segment_df.get('email', pd.Series()).notna().sum()
                has_phone = segment_df.get('phone', pd.Series()).notna().sum()
                has_name = segment_df.get('fn', pd.Series()).notna().sum()
                st.write(f"📧 Email: {has_email:,} | 📱 Phone: {has_phone:,} | 👤 Name: {has_name:,}")
            
            st.dataframe(segment_df.head(20), use_container_width=True)


def export_section(segments: Dict[str, pd.DataFrame]):
    """Display export options and provide download buttons."""
    st.markdown("### 📦 Export Audiences")
    
    hash_meta = st.checkbox(
        "🔒 Hash personally identifiable fields for Meta",
        value=True,
        help="SHA-256 hash PII fields for privacy (recommended)"
    )
    
    for segment_name, segment_df in segments.items():
        with st.expander(f"📁 {segment_name} ({len(segment_df):,} rows)", expanded=True):
            # Generate exports
            meta_df = export_meta.export_meta(segment_df, hash_fields=hash_meta)
            google_df = export_google.export_google(segment_df)
            tiktok_df = export_tiktok.export_tiktok(segment_df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.download_button(
                    label="⬇️ Download Meta CSV",
                    data=meta_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{slugify(segment_name)}_meta.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"Meta format • {len(meta_df)} rows")
            
            with col2:
                st.download_button(
                    label="⬇️ Download Google CSV",
                    data=google_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{slugify(segment_name)}_google.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"Google format • {len(google_df)} rows")
            
            with col3:
                st.download_button(
                    label="⬇️ Download TikTok CSV",
                    data=tiktok_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{slugify(segment_name)}_tiktok.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.caption(f"TikTok format • {len(tiktok_df)} rows")


# Main application flow
def main():
    # Sidebar with instructions
    with st.sidebar:
        st.markdown("### 📖 Quick Guide")
        st.markdown("""
        1. **Upload** your customer list (CSV/XLSX)
        2. **Map** columns to standard fields
        3. **Review** cleaning results
        4. **Enrich** with additional data
        5. **Segment** your audience
        6. **Export** for Meta, Google, or TikTok
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Include **email** or **phone** for best match rates
        - **Names** improve targeting precision
        - **Location** data enables geo-targeting
        - Use **A/B splits** for campaign testing
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        show_debug = st.checkbox("Show debug info", value=False)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📤 Upload your customer list",
        type=["csv", "xlsx"],
        help="Drag and drop or browse for a CSV or Excel file"
    )
    
    if uploaded_file:
        # Load file
        with st.spinner("Loading file..."):
            raw_df = load_uploaded_file(uploaded_file)
        
        st.success(f"✅ Loaded {len(raw_df):,} rows from **{uploaded_file.name}**")
        
        # Show raw preview
        with st.expander("👀 Raw Data Preview", expanded=False):
            st.dataframe(raw_df.head(10), use_container_width=True)
        
        # Field mapping
        mapping = field_mapping_ui(raw_df)
        
        if st.button("🚀 Process Audience", type="primary", use_container_width=True):
            with st.spinner("Processing your audience..."):
                # Apply mapping
                mapped_df = apply_mapping(raw_df, mapping)
                
                # Clean data
                cleaned_df, stats, summary = cleaning.clean_dataframe(mapped_df)
                
                # Display results
                display_cleaning_stats(stats, summary)
                
                # Enrichment
                enriched_df = enrichment_section(cleaned_df)
                
                # Segmentation
                segments = segmentation_section(enriched_df)
                
                # Preview
                preview_section(segments)
                
                # Export
                export_section(segments)
                
                # Success message
                st.balloons()
                st.success("🎉 Audience processing complete! Your files are ready for download above.")
                
                if show_debug:
                    with st.expander("🔍 Debug Information"):
                        st.write("**Final DataFrame Info:**")
                        st.write(f"Shape: {enriched_df.shape}")
                        st.write(f"Columns: {list(enriched_df.columns)}")
                        st.write("**Sample Data:**")
                        st.dataframe(enriched_df.head())


if __name__ == "__main__":
    main()
