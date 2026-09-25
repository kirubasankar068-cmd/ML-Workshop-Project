import pandas as pd
import numpy as np
import os
from src.discovery import perform_discovery, raw_audit
from src.quality_rules import audit_service_ratings, audit_delays, audit_categoricals, audit_duplicates
from src.cleaning import clean_dataset

def run_pipeline():
    # Paths
    train_path = 'docs/airline-satisfaction-analysis-train.csv'
    test_path = 'docs/airline-satisfaction-analysis-test.csv'

    print("--- Phase 1: Dataset Discovery ---")
    meta, compatible, df_train_raw, df_test_raw = perform_discovery(train_path, test_path)
    print(f"Schema Compatible: {compatible}")
    print(f"Train: {meta['train']['rows']} rows, {meta['train']['cols']} cols")
    print(f"Test: {meta['test']['rows']} rows, {meta['test']['cols']} cols")

    print("\n--- Phase 2 & 3: Raw Data Audit ---")
    train_audit_raw = raw_audit(df_train_raw, "Train")
    test_audit_raw = raw_audit(df_test_raw, "Test")

    train_ratings = audit_service_ratings(df_train_raw)
    train_delays = audit_delays(df_train_raw)
    train_cats = audit_categoricals(df_train_raw)
    train_dupes = audit_duplicates(df_train_raw)

    test_ratings = audit_service_ratings(df_test_raw)
    test_delays = audit_delays(df_test_raw)
    test_cats = audit_categoricals(df_test_raw)
    test_dupes = audit_duplicates(df_test_raw)

    # Summary for User (Simplified)
    print(f"Train Missing Values: {df_train_raw.isnull().sum().sum()}")
    print(f"Train Duplicates: {train_dupes['duplicates']}")

    print("\n--- Phase 4 & 5: Data Cleaning & Audit Trail ---")
    log_df = pd.DataFrame(columns=[
        'timestamp', 'dataset', 'column', 'issue_detected',
        'original_condition', 'action_taken', 'reason',
        'rows_affected', 'values_changed', 'output_column_state'
    ])

    df_train_clean, log_df = clean_dataset(df_train_raw, "train", log_df)
    df_test_clean, log_df = clean_dataset(df_test_raw, "test", log_df)

    # Save cleaned files
    os.makedirs('data/processed', exist_ok=True)
    df_train_clean.to_csv('data/processed/train_clean.csv', index=False)
    df_test_clean.to_csv('data/processed/test_clean.csv', index=False)

    # Save audit log
    os.makedirs('data/audit', exist_ok=True)
    log_df.to_csv('data/audit/data_cleaning_log.csv', index=False)

    print("Cleaned files and audit log saved successfully.")

    # Phase 6: Before vs After Comparison
    print("\n--- Phase 6: Before vs After Comparison ---")
    metrics = ['rows', 'cols', 'missing', 'duplicates']
    comparison = []

    for name, raw, clean in [("Train", df_train_raw, df_train_clean), ("Test", df_test_raw, df_test_clean)]:
        comparison.append({
            "Dataset": name,
            "Rows_Before": raw.shape[0], "Rows_After": clean.shape[0],
            "Cols_Before": raw.shape[1], "Cols_After": clean.shape[1],
            "Missing_Before": raw.isnull().sum().sum(), "Missing_After": clean.isnull().sum().sum(),
            "Duplicates_Before": raw.duplicated().sum(), "Duplicates_After": clean.duplicated().sum(),
        })

    comp_df = pd.DataFrame(comparison)
    print(comp_df)

    return df_train_clean, df_test_clean, log_df

if __name__ == "__main__":
    run_pipeline()
