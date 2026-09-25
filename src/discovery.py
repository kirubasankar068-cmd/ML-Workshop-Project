import pandas as pd
import numpy as np
import os

def perform_discovery(train_path, test_path):
    """
    Phase 1: Dataset Discovery
    Loads datasets and returns basic metadata.
    """
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    metadata = {
        "train": {
            "rows": df_train.shape[0],
            "cols": df_train.shape[1],
            "columns": df_train.columns.tolist(),
            "dtypes": df_train.dtypes.to_dict(),
            "head": df_train.head(5)
        },
        "test": {
            "rows": df_test.shape[0],
            "cols": df_test.shape[1],
            "columns": df_test.columns.tolist(),
            "dtypes": df_test.dtypes.to_dict(),
            "head": df_test.head(5)
        }
    }

    schema_compatible = set(df_train.columns) == set(df_test.columns)

    return metadata, schema_compatible, df_train, df_test

def raw_audit(df, dataset_name="Dataset"):
    """
    Phase 2: Raw Data Audit
    Calculates comprehensive statistics for a dataframe.
    """
    audit_results = {}

    # Basic Shape
    audit_results['shape'] = df.shape

    # Missing Values
    missing = df.isnull().sum()
    audit_results['missing_count'] = missing.to_dict()
    audit_results['missing_pct'] = (missing / len(df) * 100).to_dict()

    # Duplicates
    audit_results['duplicates'] = df.duplicated().sum()
    audit_results['duplicates_pct'] = (df.duplicated().sum() / len(df) * 100)

    # Numerical Analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    audit_results['numeric_stats'] = df[numeric_cols].describe().to_dict()

    # Zeros and Negatives
    audit_results['zero_counts'] = (df[numeric_cols] == 0).sum().to_dict()
    audit_results['negative_counts'] = (df[numeric_cols] < 0).sum().to_dict()

    # Categorical Analysis
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    audit_results['unique_values'] = {col: df[col].unique().tolist() for col in cat_cols}
    audit_results['unique_counts'] = df[cat_cols].nunique().to_dict()

    # Target Distribution (if satisfaction exists)
    if 'satisfaction' in df.columns:
        audit_results['target_dist'] = df['satisfaction'].value_counts(normalize=True).to_dict()

    return audit_results

if __name__ == "__main__":
    train_path = 'docs/airline-satisfaction-analysis-train.csv'
    test_path = 'docs/airline-satisfaction-analysis-test.csv'

    meta, compatible, train, test = perform_discovery(train_path, test_path)
    print(f"Schema Compatible: {compatible}")
    print(f"Train Shape: {meta['train']['rows']}x{meta['train']['cols']}")
    print(f"Test Shape: {meta['test']['rows']}x{meta['test']['cols']}")

    train_audit = raw_audit(train, "Train")
    test_audit = raw_audit(test, "Test")

    print("Audit complete. Detailed stats calculated.")
