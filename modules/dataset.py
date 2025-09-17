import streamlit as st


def run(df1):
    # show dataset
    st.markdown("")

    st.markdown("### 🗐 Dataset Shopping")
    st.dataframe(df1)