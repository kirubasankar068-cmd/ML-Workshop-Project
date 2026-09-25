import pandas as pd
import numpy as np
import datetime

def get_service_rating_cols(df):
    """Identifies the 14 service rating columns."""
    return [
        'Inflight wifi service', 'Departure/Arrival time convenient',
        'Ease of Online booking', 'Gate location', 'Food and drink',
        'Online boarding', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling',
        'Checkin service', 'Inflight service', 'Cleanliness'
    ]

def audit_service_ratings(df):
    """Phase 3A: Service Rating Audit."""
    rating_cols = get_service_rating_cols(df)
    results = {}
    for col in rating_cols:
        if col in df.columns:
            zeros = (df[col] == 0).sum()
            pct_zeros = (zeros / len(df)) * 100
            out_of_range = ((df[col] < 0) | (df[col] > 5)).sum()
            missing = df[col].isnull().sum()
            results[col] = {
                'zeros': zeros,
                'pct_zeros': pct_zeros,
                'out_of_range': out_of_range,
                'missing': missing
            }
    return results

def audit_delays(df):
    """Phase 3B: Delay Variable Audit."""
    delay_cols = ['Departure Delay in Minutes', 'Arrival Delay in Minutes']
    results = {}
    for col in delay_cols:
        if col in df.columns:
            results[col] = {
                'missing': df[col].isnull().sum(),
                'zeros': (df[col] == 0).sum(),
                'negatives': (df[col] < 0).sum(),
                'max': df[col].max(),
                'min': df[col].min(),
                'median': df[col].median(),
                'pct_25': df[col].quantile(0.25),
                'pct_75': df[col].quantile(0.75),
                'pct_95': df[col].quantile(0.95),
                'pct_99': df[col].quantile(0.99),
            }

    if all(col in df.columns for col in delay_cols):
        results['correlation'] = df[delay_cols].corr().iloc[0, 1]

    return results

def audit_categoricals(df):
    """Phase 3C: Categorical Consistency Audit."""
    cols_to_check = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    results = {}
    for col in cols_to_check:
        if col in df.columns:
            results[col] = df[col].unique().tolist()
    return results

def audit_duplicates(df):
    """Phase 3D: Duplicate Audit."""
    dupes = df.duplicated().sum()
    pct_dupes = (dupes / len(df)) * 100

    # Check if ID can distinguish them
    has_id = 'id' in df.columns
    id_dupes = df['id'].duplicated().sum() if has_id else "N/A"

    return {
        'duplicates': dupes,
        'pct_duplicates': pct_dupes,
        'id_duplicates': id_dupes
    }
