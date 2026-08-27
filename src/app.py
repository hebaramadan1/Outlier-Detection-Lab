import streamlit as st
import pandas as pd
import plotly.express as px
from stats_engine import StatisticalDetector

# Page Configuration
st.set_page_config(
    page_title="Interactive Outlier Detection Lab",
    page_icon="📊",
    layout="wide"
)

# Main Header
st.markdown("### 📊 Interactive Outlier Detection & Treatment Lab")
st.markdown("An intelligent, high-performance laboratory designed to analyze, visualize, and neutralize anomalies seamlessly.")
st.divider()

# Creating Tabs for Navigation
tab_practical, tab_bulk, tab_theory = st.tabs([
    "Single Feature Deep-Dive", 
    "Full Dataset Batch Studio", 
    "Theoretical Research Guide"
])

# ==================== TAB 1: SINGLE FEATURE DEEP-DIVE ====================
with tab_practical:
    st.sidebar.markdown("### Interactive Control Panel")
    uploaded_file = st.sidebar.file_uploader("Upload Custom Dataset (CSV)", type=["csv"], key="single_upload")

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("Dataset loaded successfully.")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    else:
        # محاولة قراءة الملف من المسارات المحتملة تلقائياً
        for path in ["heart.csv", "src/heart.csv", "../heart.csv", "data/heart.csv", "src/data/heart.csv"]:
            try:
                df = pd.read_csv(path)
                st.sidebar.info("Default Active: Heart Disease Dataset")
                break
            except:
                continue
        
        if df is None:
            st.sidebar.warning("Please upload a CSV dataset to begin.")

    if df is not None:
        with st.expander("Click to preview raw dataset matrix (First 10 Rows)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        if num_cols:
            st.sidebar.divider()
            selected_col = st.sidebar.selectbox("Select Numeric Feature", num_cols)
            detection_method = st.sidebar.selectbox("Outlier Detection Method", ["IQR Method", "Z-Score Method"])
            treatment_method = st.sidebar.selectbox("Treatment Strategy", ["None", "Trimming (Delete)", "Winsorization (Capping)", "Imputation (Median)"])

            if selected_col:
                st.markdown(f"#### Real-Time Analysis for Feature: `{selected_col}`")
                
                detector = StatisticalDetector(df)

                lower_fence, upper_fence = None, None
                if detection_method == "IQR Method":
                    result = detector.detect_outliers_iqr(selected_col)
                    outliers = result["outliers"]
                    lower_fence = result["lower_fence"]
                    upper_fence = result["upper_fence"]
                else:
                    result = detector.detect_outliers_zscore(selected_col)
                    outliers = result["outliers"]
                    lower_fence = df[selected_col].mean() - 3 * df[selected_col].std()
                    upper_fence = df[selected_col].mean() + 3 * df[selected_col].std()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Total Records", value=len(df))
                with col2:
                    st.metric(label="Outliers Detected", value=len(outliers), delta=f"-{len(outliers)} rows", delta_color="inverse")
                with col3:
                    st.metric(label="Outlier Percentage", value=f"{(len(outliers)/len(df))*100:.2f}%")

                st.divider()
                
                st.markdown("#### Distribution State Before Treatment")
                fig_before = px.box(df, y=selected_col, title=f"Boxplot for {selected_col} (Original Distribution)", color_discrete_sequence=['#2563eb'])
                fig_before.update_layout(transition_duration=600)
                st.plotly_chart(fig_before, use_container_width=True)
                
                if treatment_method != "None":
                    st.divider()
                    st.markdown(f"#### Distribution State After Treatment: `{treatment_method}`")
                    
                    if treatment_method == "Trimming (Delete)":
                        df_treated = detector.trim_outliers(
                            selected_col,
                            method="iqr" if detection_method == "IQR Method" else "zscore"
                        )

                    elif treatment_method == "Winsorization (Capping)":
                        df_treated = detector.winsorize_outliers(
                            selected_col,
                            method="iqr" if detection_method == "IQR Method" else "zscore"
                        )

                    elif treatment_method == "Imputation (Median)":
                        df_treated = detector.impute_outliers(
                            selected_col,
                            method="iqr" if detection_method == "IQR Method" else "zscore",
                            strategy="median"
                        )
                    
                    fig_after = px.box(df_treated, y=selected_col, title=f"Boxplot for {selected_col} (Cleaned Distribution)", color_discrete_sequence=['#059669'])
                    fig_after.update_layout(transition_duration=600)
                    st.plotly_chart(fig_after, use_container_width=True)
                    
                    col_btn1, col_btn2 = st.columns([1, 2])
                    with col_btn1:
                        csv_data = df_treated.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Cleaned CSV",
                            data=csv_data,
                            file_name="cleaned_feature_dataset.csv",
                            mime="text/csv",
                            type="primary"
                        )
        else:
            st.warning("The uploaded dataset does not contain numeric columns for analysis.")

# ==================== TAB 2: FULL DATASET BATCH STUDIO ====================
with tab_bulk:
    st.markdown("### Full Dataset Batch Studio")
    st.markdown("Scan and neutralize anomalies across all numeric features simultaneously with batch operations.")
    
    if df is not None:
        bulk_detection = st.selectbox("Select Batch Detection Algorithm", ["IQR Method", "Z-Score Method"], key="bulk_det")
        
        st.markdown("#### Comprehensive Outliers Summary Matrix")

        detector = StatisticalDetector(df)
        summary_data = []

        for column in df.select_dtypes(include=['number']).columns:
            if bulk_detection == "IQR Method":
                result = detector.detect_outliers_iqr(column)
            else:
                result = detector.detect_outliers_zscore(column)

            summary_data.append({
                "Feature": column,
                "Outliers": result["count"],
                "Outlier %": (result["count"] / len(df)) * 100
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        st.divider()
        st.markdown("#### Apply Global Batch Treatment")
        bulk_treatment = st.selectbox("Select Global Treatment Method", ["Trimming (Delete)", "Winsorization (Capping)", "Imputation (Median)"], key="bulk_treat")
        
        if st.button("Execute Global Batch Cleaning", type="primary"):
            with st.spinner("Processing dataset matrix... Please wait"):
                df_fully_cleaned = df.copy()

                for column in df.select_dtypes(include=['number']).columns:
                    detector = StatisticalDetector(df_fully_cleaned)

                    method = "iqr" if bulk_detection == "IQR Method" else "zscore"

                    if bulk_treatment == "Trimming (Delete)":
                        df_fully_cleaned = detector.trim_outliers(
                            column,
                            method=method
                        )

                    elif bulk_treatment == "Winsorization (Capping)":
                        df_fully_cleaned = detector.winsorize_outliers(
                            column,
                            method=method
                        )

                    elif bulk_treatment == "Imputation (Median)":
                        df_fully_cleaned = detector.impute_outliers(
                            column,
                            method=method,
                            strategy="median"
                        )
            
            st.balloons()
            st.success(f"Batch processing completed successfully. Original rows: {len(df)} | Cleaned rows: {len(df_fully_cleaned)}")
            
            full_csv_data = df_fully_cleaned.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Fully Cleaned Dataset Matrix (CSV)",
                data=full_csv_data,
                file_name="fully_cleaned_dataset.csv",
                mime="text/csv",
                type="primary"
            )

# ==================== TAB 3: THEORETICAL RESEARCH GUIDE ====================
with tab_theory:
    st.markdown("### Comprehensive Research Guide on Outliers")
    st.markdown("Explore the theoretical framework and mathematical principles powering this laboratory environment.")
    
    st.markdown("""
    ### 1. What is an Outlier?
    An outlier is an observation that lies an abnormal distance from other values in a random sample from a population. Addressing anomalies properly ensures robust machine learning models and prevents statistical distortion.
    
    ### 2. The Interquartile Range (IQR) Methodology
    A non-parametric approach relying on data dispersion:
    - **Q1 & Q3:** The 25th and 75th percentiles.
    - **IQR Formula:** `IQR = Q3 - Q1`
    - **Fences:** Lower fence `Q1 - 1.5 * IQR` | Upper fence `Q3 + 1.5 * IQR`

    ### 3. Advanced Treatment Strategies
    - **Trimming:** Deletes outlier records entirely. Ideal for large datasets with entry errors.
    - **Winsorization:** Caps extreme values at calculated fences, preserving row counts.
    - **Imputation:** Replaces anomalies with statistical metrics like median values.
    """)