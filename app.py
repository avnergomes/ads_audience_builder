import streamlit as st
import pandas as pd
from utils import cleaning, enrich, export_meta, export_google, export_tiktok, segmentation

st.set_page_config(page_title="Smart Custom Audience Builder", layout="wide")

st.title("🧠 Smart Custom Audience Builder")
st.write("Upload, clean, enrich, and export your audience lists for Meta, Google, and TikTok Ads.")

uploaded_file = st.file_uploader("📤 Upload a customer list (CSV/XLSX)", type=["csv", "xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    st.write("Data preview loaded successfully.")
    # Future: cleaning, enrichment, export logic hooks go here
