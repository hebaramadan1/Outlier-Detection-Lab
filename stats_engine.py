import pandas as pd
import numpy as np

def detect_iqr_outliers(df, column, multiplier=1.5):
    """Detect outliers using the Interquartile Range method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - multiplier * IQR
    upper_fence = Q3 + multiplier * IQR
    
    outliers = df[(df[column] < lower_fence) | (df[column] > upper_fence)]
    return outliers, lower_fence, upper_fence

def detect_zscore_outliers(df, column, threshold=3.0):
    """Detect outliers using the Z-Score method."""
    mean = df[column].mean()
    std = df[column].std()
    if std == 0:
        return pd.DataFrame()
    z_scores = np.abs((df[column] - mean) / std)
    outliers = df[z_scores > threshold]
    return outliers

def treat_outliers(df, column, treatment_method, lower_fence, upper_fence):
    """Treat outliers in a specific column based on the chosen strategy."""
    df_treated = df.copy()
    
    if treatment_method == "Trimming (Delete)":
        df_treated = df_treated[(df_treated[column] >= lower_fence) & (df_treated[column] <= upper_fence)]
    elif treatment_method == "Winsorization (Capping)":
        df_treated[column] = df_treated[column].clip(lower=lower_fence, upper=upper_fence)
    elif treatment_method == "Imputation (Median)":
        median_val = df_treated[column].median()
        mask = (df_treated[column] < lower_fence) | (df_treated[column] > upper_fence)
        df_treated.loc[mask, column] = median_val
        
    return df_treated

def get_all_columns_summary(df, detection_method="IQR Method"):
    """Generate a summary dataframe for outliers across all numeric columns."""
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    summary_data = []
    
    for col in num_cols:
        if df[col].nunique() <= 4:
            continue
            
        if detection_method == "IQR Method":
            outliers, _, _ = detect_iqr_outliers(df, col)
        else:
            outliers = detect_zscore_outliers(df, col)
            
        count = len(outliers)
        percentage = (count / len(df)) * 100
        
        summary_data.append({
            "Feature Name": col,
            "Outliers Count": count,
            "Outlier Percentage (%)": round(percentage, 2)
        })
        
    return pd.DataFrame(summary_data)

def treat_all_numeric_columns(df, treatment_method, detection_method="IQR Method"):
    """Apply global batch treatment across all numeric columns."""
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    
    for col in num_cols:
        if df_clean[col].nunique() <= 4:
            continue
            
        if detection_method == "IQR Method":
            _, lower_fence, upper_fence = detect_iqr_outliers(df_clean, col)
        else:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            lower_fence = mean - 3 * std
            upper_fence = mean + 3 * std
            
        df_clean = treat_outliers(df_clean, col, treatment_method, lower_fence, upper_fence)
        
    return df_clean