# Airline Passenger Data Quality Report

## 1. Dataset Overview
The airline passenger satisfaction dataset consists of two files:
- **Training Set**: 103,904 records, 25 columns.
- **Testing Set**: 25,976 records, 25 columns.

The goal is to analyze passenger demographics and travel experience to optimize a ₹500 crore investment.

## 2. Train/Test Schema
- **Compatibility**: Fully compatible.
- **Common Columns**: 25 columns across both sets.
- **Structure**: Mix of numerical (Age, Flight Distance, Delays, Ratings) and categorical (Gender, Customer Type, Travel Type, Class) data.

## 3. Missing Value Audit
- **Initial State**: 310 missing values detected in the training set.
- **Treatment**: 
    - Numerical columns were imputed using the **Median**.
    - Categorical columns were imputed using the **Mode**.
- **Final State**: 0 missing values in both training and testing sets.

## 4. Duplicate Audit
- **Exact Duplicates**: 0 duplicates found in both raw and cleaned datasets.
- **ID Uniqueness**: The `id` column is unique; no duplicate passenger records were found.

## 5. Service Rating Audit
The 14 service-rating columns (e.g., `Inflight wifi service`, `Cleanliness`) were audited:
- **Issue**: Significant number of `0` values (e.g., 5,300 in `Departure/Arrival time convenient`).
- **Action**: All `0` values were converted to `NaN` and subsequently imputed.
- **Reason**: On a 1-5 scale, `0` represents "Not Applicable" rather than a rating of zero.
- **Result**: Final ranges for all rating columns are strictly [1.0, 5.0].

## 6. Delay Audit
- **Columns**: `Departure Delay in Minutes` and `Arrival Delay in Minutes`.
- **Outliers**: Approximately 13-14% of records are IQR outliers (Max values ~1,590 minutes).
- **Decision**: **Deliberately preserved**. Extreme delays are critical business indicators for investment analysis.
- **Modification**: No delay values were changed or capped.

## 7. Categorical Audit
- **Columns**: `Gender`, `Customer Type`, `Type of Travel`, `Class`, `satisfaction`.
- **Consistency**: Checked for whitespace and casing.
- **Action**: Applied `.strip()` to categorical columns.
- **Observation**: No merged categories were necessary as no duplicates were found.

## 8. Outlier Audit
- **Numerical Outliers**: Present in `Flight Distance` and `Delay` columns.
- **Treatment**: Preserved for business intelligence purposes. No clipping or removal was performed.

## 9. Target Audit
- **Target**: `satisfaction`
- **Values**: `satisfied` (43.3%) and `neutral or dissatisfied` (56.7%).
- **Integrity**: No target imputation was performed. The target was not used in any cleaning orpreprocessing step.

## 10. Cleaning Decisions

| What Changed? | Why? | Rows Affected | Risk if not changed? |
| :--- | :--- | :--- | :--- |
| Removed `Unnamed: 0` | Meaningless index | 103,904 | Noise in future ML models |
| 0 $\rightarrow$ NaN (Ratings) | 0 is "N/A", not a rating | ~15,000+ | skewed mean/median ratings |
| Missing $\rightarrow$ Imputed | Ensure data completeness | 310 | Loss of passenger data |
| Whitespace Strip | Data consistency | Various | Category duplication (e.g. " Male" vs "Male") |

## 11. Before vs After

| Metric | Before (Train) | After (Train) | Change |
| :--- | :--- | :--- | :--- |
| Rows | 103,904 | 103,904 | None |
| Columns | 25 | 24 | -1 (`Unnamed: 0`) |
| Missing Values | 310 | 0 | -310 |
| Duplicates | 0 | 0 | None |
| Rating Zeros | ~15,000 | 0 | All converted/imputed |

## 12. Remaining Issues
No critical data quality issues remain. The data is now standardized and complete.

## 13. ML Readiness
The dataset is ready for Feature Engineering.
