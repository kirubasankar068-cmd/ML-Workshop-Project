import pandas as pd
import numpy as np
import datetime
import os
from src.quality_rules import get_service_rating_cols

def log_cleaning_action(log_df, dataset, column, issue, original, action, reason, rows, values, state):
    """Helper to append actions to the audit log."""
    new_entry = {
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset': dataset,
        'column': column,
        'issue_detected': issue,
        'original_condition': original,
        'action_taken': action,
        'reason': reason,
        'rows_affected': rows,
        'values_changed': values,
        'output_column_state': state
    }
    return pd.concat([log_df, pd.DataFrame([new_entry])], ignore_index=True)

def clean_dataset(df, dataset_name, log_df):
    """
    Phase 4: Data Cleaning
    Performs justified transformations and updates the audit log.
    """
    df_clean = df.copy()

    # 1. Remove accidental index columns
    if 'Unnamed: 0' in df_clean.columns:
        rows_affected = len(df_clean)
        df_clean = df_clean.drop(columns=['Unnamed: 0'])
        log_df = log_cleaning_action(
            log_df, dataset_name, 'Unnamed: 0', 'Accidental index column',
            'Column exists', 'Dropped column', 'Meaningless index',
            rows_affected, 'N/A', 'Removed'
        )

    # 2. Strip whitespace and standardize categoricals
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        # Detect if whitespace exists
        whitespace_exists = df_clean[col].astype(str).str.contains(r'^\s+|\s+$').any()
        if whitespace_exists:
            rows_affected = df_clean[col].astype(str).str.contains(r'^\s+|\s+$').sum()
            df_clean[col] = df_clean[col].astype(str).str.strip()
            log_df = log_cleaning_action(
                log_df, dataset_name, col, 'Leading/trailing whitespace',
                'Spaces present', 'Applied .strip()', 'Data consistency',
                rows_affected, 'Trimmed', 'Cleaned'
            )

    # 3. Convert service-rating zeros to NaN
    rating_cols = get_service_rating_cols(df_clean)
    for col in rating_cols:
        if col in df_clean.columns:
            zeros = (df_clean[col] == 0).sum()
            if zeros > 0:
                df_clean[col] = df_clean[col].replace(0, np.nan)
                log_df = log_cleaning_action(
                    log_df, dataset_name, col, 'Zero values in rating',
                    '0 represents Not Applicable', 'Converted 0 to NaN',
                    'Ratings should be 1-5', zeros, '0 -> NaN', 'NaN-filled'
                )

    # 4. Handle missing values (Simple Imputation for now, will be refined after audit)
    # Numeric: Median, Categorical: Mode
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            log_df = log_cleaning_action(
                log_df, dataset_name, col, 'Missing values',
                f'{missing} NaNs', 'Median imputation', 'Preserve observations',
                missing, f'NaN -> {median_val}', 'Imputed'
            )

    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            log_df = log_cleaning_action(
                log_df, dataset_name, col, 'Missing values',
                f'{missing} NaNs', 'Mode imputation', 'Preserve observations',
                missing, f'NaN -> {mode_val}', 'Imputed'
            )

    return df_clean, log_df

if __name__ == "__main__":
    # This is a module, normally called by main.py
    pass
