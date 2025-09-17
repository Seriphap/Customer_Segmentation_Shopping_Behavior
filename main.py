import streamlit as st
import pandas as pd
from modules import dataset, Clustering

st.set_page_config(layout="wide")
st.markdown("##  👨‍👩‍👦‍👦 Customer and Marketing Analytics - Clustering")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type=["csv"])
st.text("Note: Please ensure your uploaded dataset follows the same format and structure as the default example dataset to avoid errors during processing")

# Load dataset
if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
else:
    df1 = pd.read_csv("data/8_Shopping.csv")
    st.info("ℹ️ Using default dataset")

# Sidebar
st.sidebar.markdown("## 🔍 Select Section")
menu = st.sidebar.radio("", ["🗐 Dataset", "🛒 Clustering"])

if menu == "🗐 Dataset":
    dataset.run(df1)
elif menu == "🛒 Clustering":
    Clustering.run(df1)



