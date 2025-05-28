import streamlit as st
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(
    page_title="AIC",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("AIC")
st.markdown("""
This application provides AI-powered consulting services using advanced analytics and machine learning.
""")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select a page:",
        ["Dashboard", "Analysis", "Reports"]
    )

# Main content
if page == "Dashboard":
    st.header("Dashboard")
    st.write("Welcome to the AI Consulting System dashboard!")
    
    # Sample metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Active Projects", value="5")
    with col2:
        st.metric(label="Completed Analysis", value="12")
    with col3:
        st.metric(label="Success Rate", value="94%")

elif page == "Analysis":
    st.header("Analysis")
    st.write("Upload your data for analysis:")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of your data:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

elif page == "Reports":
    st.header("Reports")
    st.write("Generated reports will appear here.")
    
    # Sample report selection
    report_type = st.selectbox(
        "Select report type:",
        ["Business Analysis", "Technical Overview", "Executive Summary"]
    )
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            st.success("Report generated successfully!")
            st.download_button(
                label="Download Report",
                data="Sample report content",
                file_name=f"{report_type.lower().replace(' ', '_')}.txt"
            )
