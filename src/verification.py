import pandas as pd
import numpy as np
import os

def verify_pipeline():
    # Paths
    train_raw_path = 'docs/airline-satisfaction-analysis-train.csv'
    test_raw_path = 'docs/airline-satisfaction-analysis-test.csv'
    train_clean_path = 'data/processed/train_clean.csv'
    test_clean_path = 'data/processed/test_clean.csv'

    df_train_raw = pd.read_csv(train_raw_path)
    df_test_raw = pd.read_csv(test_raw_path)
    df_train_clean = pd.read_csv(train_clean_path)
    df_test_clean = pd.read_csv(test_clean_path)

    print("--- 1. File Existence Check ---")
    files_to_check = [
        'data/processed/train_clean.csv',
        'data/processed/test_clean.csv',
        'data/audit/data_cleaning_log.csv',
        'data/audit/data_audit_report.csv',
        'reports/data_quality_report.md'
    ]
    for f in files_to_check:
        exists = os.path.exists(f)
        print(f"{f}: {'EXISTS' if exists else 'MISSING'}")

    print("\n--- 2. Raw Data Integrity Check ---")
    # Since we have the raw files, we just check if their shapes match what was reported in discovery
    print(f"Train Raw Shape: {df_train_raw.shape} (Expected 103904 x 25)")
    print(f"Test Raw Shape: {df_test_raw.shape} (Expected 25976 x 25)")

    print("\n--- 3. Service Rating Verification ---")
    rating_cols = [
        'Inflight wifi service', 'Departure/Arrival time convenient',
        'Ease of Online booking', 'Gate location', 'Food and drink',
        'Online boarding', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Inflight service', 'Cleanliness'
    ]

    for col in rating_cols:
        raw_zeros = (df_train_raw[col] == 0).sum()
        clean_zeros = (df_train_clean[col] == 0).sum()
        raw_missing = df_train_raw[col].isnull().sum()
        clean_missing = df_train_clean[col].isnull().sum()

        print(f"{col}:")
        print(f"  Raw Zeros: {raw_zeros} | Clean Zeros: {clean_zeros}")
        print(f"  Raw Missing: {raw_missing} | Clean Missing: {clean_missing}")
        print(f"  Clean Range: [{df_train_clean[col].min()}, {df_train_clean[col].max()}]")

    print("\n--- 4. Delay Outlier Verification ---")
    delay_cols = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
    for col in delay_cols:
        raw_stats = df_train_raw[col].describe(percentiles=[.25, .5, .75, .95, .99])
        clean_stats = df_train_clean[col].describe(percentiles=[.25, .5, .75, .95, .99])

        print(f"{col}:")
        print(f"  Raw Max: {raw_stats['max']} | Clean Max: {clean_stats['max']}")
        print(f"  Raw Min: {raw_stats['min']} | Clean Min: {clean_stats['min']}")

        # IQR calculation
        q1, q3 = raw_stats['25%'], raw_stats['75%']
        iqr = q3 - q1
        outliers = ((df_train_raw[col] < (q1 - 1.5 * iqr)) | (df_train_raw[col] > (q3 + 1.5 * iqr))).sum()
        print(f"  IQR Outliers (Raw): {outliers} ({outliers/len(df_train_raw)*100:.2f}%)")

    print("\n--- 5. Missing Value Handling Verification ---")
    print("Train Dataset:")
    print(f"Missing Before: {df_train_raw.isnull().sum().sum()}")
    print(f"Missing After: {df_train_clean.isnull().sum().sum()}")

    print("\n--- 6. Categorical Consistency Verification ---")
    cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    for col in cat_cols:
        print(f"{col} Raw: {df_train_raw[col].unique()}")
        print(f"{col} Clean: {df_train_clean[col].unique()}")

    print("\n--- 7. Duplicate Verification ---")
    print(f"Raw Duplicates: {df_train_raw.duplicated().sum()}")
    print(f"Clean Duplicates: {df_train_clean.duplicated().sum()}")
    print(f"Raw ID Duplicates: {df_train_raw['id'].duplicated().sum()}")

    print("\n--- 8. Target Variable Verification ---")
    print(f"Target Column: satisfaction")
    print(f"Unique Values: {df_train_clean['satisfaction'].unique()}")
    print(f"Missing Values: {df_train_clean['satisfaction'].isnull().sum()}")
    print(f"Distribution:\n{df_train_clean['satisfaction'].value_counts(normalize=True)}")

if __name__ == "__main__":
    verify_pipeline()
