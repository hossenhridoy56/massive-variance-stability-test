# ============================================================
# MVT REAL-DATA ANALYSIS
# Gold Futures Historical Data
#
# Final version:
#   - Correct MM/DD/YYYY date parsing
#   - Full data retained
#   - Log returns
#   - ADF + KPSS
#   - ARIMA model selection with convergence check
#   - MVT k-sensitivity
#   - k* = floor(c * sqrt(n)) guideline
#   - Minimum segment size constraint: m >= 50
#   - Automatic main-k selection
#   - Classical variance tests
#   - CSV outputs
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2, bartlett, levene

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_breuschpagan,
    het_white,
    het_goldfeldquandt
)

import statsmodels.api as sm


# ============================================================
# 1. SETTINGS
# ============================================================

FILE_PATH = r"C:\Users\HRIDOY\OneDrive\Desktop\MVT\Gold Futures Historical Data.csv"

ALPHA = 0.05
SEED = 12345

# Minimum observations per segment
MIN_M = 50

# c values from the proposed rule:
# k* = floor(c * sqrt(n))
C_VALUES = [2, 2.5, 3, 3.5, 4, 4.5, 5]

# Broader k-sensitivity
K_VALUES = [
    4, 5, 6, 8, 10,
    12, 15, 20, 25, 30,
    40, 50, 60, 70, 80, 90, 99
]

# Candidate ARIMA models
ARIMA_ORDERS = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 0),
    (0, 0, 2),
    (2, 0, 1),
    (1, 0, 2),
    (2, 0, 2),
    (3, 0, 0),
    (0, 0, 3),
    (3, 0, 1),
    (1, 0, 3),
    (2, 0, 3),
    (3, 0, 2),
    (3, 0, 3)
]

np.random.seed(SEED)

warnings.filterwarnings("ignore")


# ============================================================
# 2. CHECK FILE
# ============================================================

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"\nFile not found:\n{FILE_PATH}"
    )

print("=" * 75)
print("MVT REAL-DATA ANALYSIS")
print("Gold Futures Historical Data")
print("=" * 75)


# ============================================================
# 3. LOAD DATA
# ============================================================

df = pd.read_csv(FILE_PATH)

print("\nOriginal data shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 raw observations:")
print(
    df.head(10).to_string(index=False)
)

print("\nLast 10 raw observations:")
print(
    df.tail(10).to_string(index=False)
)


# ============================================================
# 4. REQUIRED COLUMNS
# ============================================================

if "Date" not in df.columns:
    raise ValueError("Date column not found.")

if "Price" not in df.columns:
    raise ValueError("Price column not found.")


# ============================================================
# 5. DATE PARSING
# ============================================================
#
# Uploaded dataset uses:
#
#       MM/DD/YYYY
#
# Example:
#
#       1/3/2005  = January 3, 2005
#       2/24/2026 = February 24, 2026
# ============================================================

raw_dates = df["Date"].copy()

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%m/%d/%Y",
    errors="coerce"
)

invalid_dates = df["Date"].isna().sum()

print("\n" + "=" * 75)
print("DATE PARSING")
print("=" * 75)

print("Total observations :", len(df))
print("Invalid dates      :", invalid_dates)
print("Valid dates        :", df["Date"].notna().sum())

if invalid_dates > 0:

    print("\nInvalid raw dates:")
    print(
        raw_dates[df["Date"].isna()]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 6. NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "Price",
    "Open",
    "High",
    "Low",
    "Vol."
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# 7. MISSING VALUE REPORT
# ============================================================

print("\n" + "=" * 75)
print("MISSING / INVALID VALUES")
print("=" * 75)

for col in [
    "Date",
    "Price",
    "Open",
    "High",
    "Low",
    "Vol."
]:

    if col in df.columns:

        print(
            f"{col:10s}: "
            f"{df[col].isna().sum()}"
        )


# ============================================================
# 8. CLEAN DATA
# ============================================================

before_cleaning = len(df)

df = df.dropna(
    subset=["Date", "Price"]
).copy()

after_cleaning = len(df)

print("\n" + "=" * 75)
print("CLEANING SUMMARY")
print("=" * 75)

print("Before cleaning :", before_cleaning)
print("After cleaning  :", after_cleaning)
print("Removed         :", before_cleaning - after_cleaning)


# ============================================================
# 9. DUPLICATE DATE CHECK
# ============================================================

duplicate_dates = df["Date"].duplicated().sum()

print("\nDuplicate dates:", duplicate_dates)

if duplicate_dates > 0:

    df = (
        df
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last"
        )
        .copy()
    )


# ============================================================
# 10. SORT CHRONOLOGICALLY
# ============================================================

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)

print("\n" + "=" * 75)
print("FINAL DATA RANGE")
print("=" * 75)

print("Start:", df["Date"].min())
print("End  :", df["Date"].max())

print(
    "Final observations:",
    len(df)
)


# ============================================================
# 11. PRICE DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 75)
print("PRICE DESCRIPTIVE STATISTICS")
print("=" * 75)

print(
    df["Price"]
    .describe()
    .to_string()
)


# ============================================================
# 12. LOG RETURNS
# ============================================================

df["Log_Return"] = (
    np.log(df["Price"])
    .diff()
    * 100
)

returns = (
    df["Log_Return"]
    .dropna()
    .reset_index(drop=True)
)

print("\n" + "=" * 75)
print("LOG RETURN SUMMARY")
print("=" * 75)

print(
    returns.describe()
    .to_string()
)

print(
    "\nSkewness:",
    returns.skew()
)

print(
    "Excess Kurtosis:",
    returns.kurt()
)


# ============================================================
# 13. FINITE VALUE CHECK
# ============================================================

if not np.isfinite(returns).all():

    raise ValueError(
        "Log returns contain NaN or infinite values."
    )


# ============================================================
# 14. PRICE PLOT
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"],
    df["Price"]
)

plt.title(
    "Gold Futures Price"
)

plt.xlabel("Date")
plt.ylabel("Price")

plt.tight_layout()

plt.savefig(
    "Gold_Price_Series.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. LOG RETURN PLOT
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    df["Date"].iloc[1:],
    returns
)

plt.axhline(
    0,
    linestyle="--"
)

plt.title(
    "Gold Futures Log Returns"
)

plt.xlabel("Date")
plt.ylabel("Log Return (%)")

plt.tight_layout()

plt.savefig(
    "Gold_Log_Returns.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. ADF TEST
# ============================================================

print("\n" + "=" * 75)
print("ADF STATIONARITY TEST")
print("=" * 75)

adf_result = adfuller(
    returns,
    autolag="AIC"
)

print(
    "ADF statistic :",
    adf_result[0]
)

print(
    "p-value       :",
    adf_result[1]
)

print(
    "Lags          :",
    adf_result[2]
)

print(
    "Observations  :",
    adf_result[3]
)


# ============================================================
# 17. KPSS TEST
# ============================================================

print("\n" + "=" * 75)
print("KPSS STATIONARITY TEST")
print("=" * 75)

try:

    kpss_result = kpss(
        returns,
        regression="c",
        nlags="auto"
    )

    print(
        "KPSS statistic :",
        kpss_result[0]
    )

    print(
        "p-value        :",
        kpss_result[1]
    )

    print(
        "Lags           :",
        kpss_result[2]
    )

except Exception as e:

    print(
        "KPSS test failed:",
        e
    )


# ============================================================
# 18. ARIMA MODEL SELECTION
# ============================================================

print("\n" + "=" * 75)
print("ARIMA MODEL SELECTION")
print("=" * 75)

arima_results = []

for order in ARIMA_ORDERS:

    print(
        f"Trying ARIMA{order} ..."
    )

    try:

        model = ARIMA(
            returns,
            order=order,
            trend="c",
            enforce_stationarity=True,
            enforce_invertibility=True
        )

        fitted = model.fit()

        converged = fitted.mle_retvals.get(
            "converged",
            True
        )

        arima_results.append({

            "p": order[0],
            "d": order[1],
            "q": order[2],

            "AIC": fitted.aic,
            "BIC": fitted.bic,
            "HQIC": fitted.hqic,

            "Converged": converged
        })

    except Exception as e:

        print(
            f"ARIMA{order} failed: {e}"
        )


arima_table = pd.DataFrame(
    arima_results
)


# ============================================================
# 19. ARIMA SELECTION TABLE
# ============================================================

if arima_table.empty:

    raise RuntimeError(
        "No ARIMA model successfully fitted."
    )

converged_models = arima_table[
    arima_table["Converged"] == True
].copy()

if converged_models.empty:

    print(
        "\nWARNING:"
        "\nNo model reported convergence."
    )

    best_row = (
        arima_table
        .sort_values("AIC")
        .iloc[0]
    )

else:

    best_row = (
        converged_models
        .sort_values("AIC")
        .iloc[0]
    )


best_order = (
    int(best_row["p"]),
    int(best_row["d"]),
    int(best_row["q"])
)


# ============================================================
# 20. SELECTED ARIMA MODEL
# ============================================================

print("\n" + "=" * 75)
print("SELECTED ARIMA MODEL")
print("=" * 75)

print(
    "Best order:",
    best_order
)

print(
    "AIC:",
    best_row["AIC"]
)

print(
    "BIC:",
    best_row["BIC"]
)

print(
    "HQIC:",
    best_row["HQIC"]
)

print(
    "Converged:",
    best_row["Converged"]
)


# ============================================================
# 21. FIT SELECTED ARIMA
# ============================================================

arima_model = ARIMA(
    returns,
    order=best_order,
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True
)

arima_fit = arima_model.fit()


print("\nARIMA summary:")
print(
    arima_fit.summary()
)


# ============================================================
# 22. ARIMA RESIDUALS
# ============================================================

residuals = (
    pd.Series(arima_fit.resid)
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .dropna()
    .reset_index(drop=True)
)

print("\n" + "=" * 75)
print("ARIMA RESIDUAL SUMMARY")
print("=" * 75)

print(
    residuals.describe()
    .to_string()
)

print(
    "\nMean:",
    residuals.mean()
)

print(
    "Variance:",
    residuals.var()
)

print(
    "Std:",
    residuals.std()
)

print(
    "Skewness:",
    residuals.skew()
)

print(
    "Excess Kurtosis:",
    residuals.kurt()
)


# ============================================================
# 23. LJUNG-BOX TEST
# ============================================================

print("\n" + "=" * 75)
print("LJUNG-BOX TEST")
print("=" * 75)

lb_result = acorr_ljungbox(
    residuals,
    lags=[20],
    return_df=True
)

print(
    lb_result.to_string()
)


# ============================================================
# 24. MVT FUNCTION
# ============================================================

def mvt_test(x, k, alpha=0.05):

    x = np.asarray(
        x,
        dtype=float
    )

    n = len(x)

    # Equal segment size
    m = n // k

    if m < 2:

        raise ValueError(
            f"Segment size m={m} is too small."
        )

    n_used = k * m

    x_used = x[:n_used]

    # k x m matrix
    segments = x_used.reshape(
        k,
        m
    )

    # Segment sample variances
    segment_variances = np.var(
        segments,
        axis=1,
        ddof=1
    )

    # Mean segment variance
    v_bar = np.mean(
        segment_variances
    )

    # Current MVT statistic
    Q = (
        (m - 1)
        /
        (2 * v_bar**2)
        *
        np.sum(
            (
                segment_variances
                - v_bar
            )**2
        )
    )

    df_mvt = k - 1

    critical_value = chi2.ppf(
        1 - alpha,
        df_mvt
    )

    p_value = chi2.sf(
        Q,
        df_mvt
    )

    return {

        "k": k,

        "m": m,

        "n_original": n,

        "n_used": n_used,

        "bar_v": v_bar,

        "Q_MVT": Q,

        "df": df_mvt,

        "p_value": p_value,

        "critical_value": critical_value,

        "reject_H0": (
            p_value < alpha
        ),

        "segment_variances":
            segment_variances
    }


# ============================================================
# 25. BROAD MVT K-SENSITIVITY
# ============================================================

print("\n" + "=" * 75)
print("MVT K-SENSITIVITY ANALYSIS")
print("=" * 75)

mvt_sensitivity = []

for k in K_VALUES:

    # Minimum segment-size restriction
    m = len(residuals) // k

    if m < MIN_M:

        print(
            f"k={k}: skipped "
            f"(m={m} < {MIN_M})"
        )

        continue

    try:

        result = mvt_test(
            residuals,
            k,
            ALPHA
        )

        mvt_sensitivity.append({

            "k": result["k"],

            "m": result["m"],

            "n_used":
                result["n_used"],

            "bar_v":
                result["bar_v"],

            "Q_MVT":
                result["Q_MVT"],

            "df":
                result["df"],

            "p_value":
                result["p_value"],

            "critical_value":
                result["critical_value"],

            "Reject_H0":
                result["reject_H0"]
        })

        print(
            f"k={result['k']:>3}, "
            f"m={result['m']:>4}, "
            f"Q={result['Q_MVT']:.6f}, "
            f"p={result['p_value']:.6e}"
        )

    except Exception as e:

        print(
            f"k={k} failed: {e}"
        )


mvt_sensitivity_df = pd.DataFrame(
    mvt_sensitivity
)


# ============================================================
# 26. k* = floor(c * sqrt(n)) GUIDELINE
# ============================================================

n_mvt = len(residuals)

sqrt_n = np.sqrt(
    n_mvt
)

print("\n" + "=" * 75)
print("K* SELECTION USING THE PROPOSED RULE")
print("=" * 75)

print(
    f"n = {n_mvt}"
)

print(
    f"sqrt(n) = {sqrt_n:.6f}"
)

kstar_rows = []

for c in C_VALUES:

    k_target = int(
        np.floor(
            c * sqrt_n
        )
    )

    m_target = n_mvt // k_target

    admissible = (
        m_target >= MIN_M
    )

    kstar_rows.append({

        "c": c,

        "k_star_target":
            k_target,

        "m_target":
            m_target,

        "Minimum_m":
            MIN_M,

        "Admissible":
            admissible
    })

    print(
        f"c={c:<3} | "
        f"k*={k_target:<3} | "
        f"m={m_target:<4} | "
        f"Admissible={admissible}"
    )


kstar_df = pd.DataFrame(
    kstar_rows
)


# ============================================================
# 27. FINAL MAIN-k SELECTION
# ============================================================
#
# The proposed c=3 rule gives a target k.
#
# If that target violates m >= MIN_M,
# choose the largest admissible k not exceeding
# the target.
#
# If no such k exists, choose the largest admissible k.
# ============================================================

c_main = 3.0

target_k = int(
    np.floor(
        c_main * sqrt_n
    )
)

admissible_k_values = [
    k
    for k in K_VALUES
    if (n_mvt // k) >= MIN_M
]


if not admissible_k_values:

    raise RuntimeError(
        "No admissible k satisfies the minimum "
        "segment-size condition."
    )


# We want the admissible k closest to c=3 target.
main_k = min(
    admissible_k_values,
    key=lambda k: abs(k - target_k)
)


# ============================================================
# 28. MAIN MVT RESULT
# ============================================================

main_result = mvt_test(
    residuals,
    main_k,
    ALPHA
)


print("\n" + "=" * 75)
print("FINAL MAIN MVT RESULT")
print("=" * 75)

print(
    f"c used                 = {c_main}"
)

print(
    f"Target k*              = {target_k}"
)

print(
    f"Selected admissible k  = {main_k}"
)

print(
    f"Segment size m         = "
    f"{main_result['m']}"
)

print(
    f"n used                 = "
    f"{main_result['n_used']}"
)

print(
    f"Mean variance          = "
    f"{main_result['bar_v']:.10f}"
)

print(
    f"Q_MVT                  = "
    f"{main_result['Q_MVT']:.10f}"
)

print(
    f"df                     = "
    f"{main_result['df']}"
)

print(
    f"p-value                = "
    f"{main_result['p_value']:.10e}"
)

print(
    f"Critical value         = "
    f"{main_result['critical_value']:.10f}"
)

print(
    f"Reject H0              = "
    f"{main_result['reject_H0']}"
)


# ============================================================
# 29. ADD MAIN RESULT TO KSTAR TABLE
# ============================================================

kstar_df["Distance_to_target"] = (
    abs(
        kstar_df["k_star_target"]
        - main_k
    )
)

kstar_df["Selected_Main_k"] = (
    kstar_df["k_star_target"]
    == target_k
)


# ============================================================
# 30. SEGMENT VARIANCES FOR MAIN k
# ============================================================

segment_variances = (
    main_result["segment_variances"]
)

segment_table = pd.DataFrame({

    "Segment":
        np.arange(
            1,
            main_k + 1
        ),

    "Variance":
        segment_variances
})


print("\n" + "=" * 75)
print("MAIN SEGMENT-WISE VARIANCES")
print("=" * 75)

print(
    segment_table.to_string(
        index=False
    )
)


# ============================================================
# 31. CLASSICAL VARIANCE TESTS
# ============================================================

print("\n" + "=" * 75)
print("CLASSICAL VARIANCE TESTS")
print("=" * 75)

n_used = main_result["n_used"]

x_used = (
    residuals
    .iloc[:n_used]
    .to_numpy()
)

segments = x_used.reshape(
    main_k,
    main_result["m"]
)


# ------------------------------------------------------------
# Bartlett
# ------------------------------------------------------------

bartlett_stat, bartlett_p = bartlett(
    *segments
)

print("\nBartlett test:")

print(
    "Statistic:",
    bartlett_stat
)

print(
    "p-value :",
    bartlett_p
)


# ------------------------------------------------------------
# Levene
# ------------------------------------------------------------

levene_stat, levene_p = levene(
    *segments,
    center="median"
)

print("\nLevene test:")

print(
    "Statistic:",
    levene_stat
)

print(
    "p-value :",
    levene_p
)


# ============================================================
# 32. BREUSCH-PAGAN TEST
# ============================================================

resid_array = x_used

squared_resid = (
    resid_array ** 2
)

time_index = np.arange(
    len(squared_resid)
)

X_bp = pd.DataFrame({

    "Time":
        time_index
})

X_bp = sm.add_constant(
    X_bp
)

bp_lm, bp_lm_p, bp_f, bp_f_p = (
    het_breuschpagan(
        squared_resid,
        X_bp
    )
)

print("\nBreusch-Pagan test:")

print(
    "LM statistic:",
    bp_lm
)

print(
    "LM p-value  :",
    bp_lm_p
)

print(
    "F statistic :",
    bp_f
)

print(
    "F p-value   :",
    bp_f_p
)


# ============================================================
# 33. WHITE TEST
# ============================================================

X_white = pd.DataFrame({

    "Time":
        time_index
})

X_white = sm.add_constant(
    X_white
)

white_lm, white_lm_p, white_f, white_f_p = (
    het_white(
        squared_resid,
        X_white
    )
)

print("\nWhite test:")

print(
    "LM statistic:",
    white_lm
)

print(
    "LM p-value  :",
    white_lm_p
)

print(
    "F statistic :",
    white_f
)

print(
    "F p-value   :",
    white_f_p
)


# ============================================================
# 34. GOLDFELD-QUANDT TEST
# ============================================================

try:

    gq_f, gq_p, gq_order = (
        het_goldfeldquandt(
            squared_resid,
            X_bp,
            alternative="two-sided"
        )
    )

    print("\nGoldfeld-Quandt test:")

    print(
        "F statistic:",
        gq_f
    )

    print(
        "p-value    :",
        gq_p
    )

    print(
        "Order      :",
        gq_order
    )

except Exception as e:

    gq_f = np.nan
    gq_p = np.nan
    gq_order = None

    print(
        "\nGoldfeld-Quandt test failed:"
    )

    print(e)


# ============================================================
# 35. COMPARISON TABLE
# ============================================================

comparison_table = pd.DataFrame({

    "Test": [

        "MVT",
        "Bartlett",
        "Levene",
        "Breusch-Pagan",
        "White",
        "Goldfeld-Quandt"
    ],

    "Statistic": [

        main_result["Q_MVT"],
        bartlett_stat,
        levene_stat,
        bp_lm,
        white_lm,
        gq_f
    ],

    "p_value": [

        main_result["p_value"],
        bartlett_p,
        levene_p,
        bp_lm_p,
        white_lm_p,
        gq_p
    ],

    "Reject_H0_5pct": [

        main_result["p_value"] < ALPHA,
        bartlett_p < ALPHA,
        levene_p < ALPHA,
        bp_lm_p < ALPHA,
        white_lm_p < ALPHA,
        gq_p < ALPHA
    ]
})


print("\n" + "=" * 75)
print("VARIANCE TEST COMPARISON")
print("=" * 75)

print(
    comparison_table.to_string(
        index=False
    )
)


# ============================================================
# 36. MVT k-SENSITIVITY PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    mvt_sensitivity_df["k"],
    mvt_sensitivity_df["Q_MVT"],
    marker="o"
)

plt.xlabel(
    "Number of Segments (k)"
)

plt.ylabel(
    "MVT Statistic"
)

plt.title(
    "MVT Statistic Across Different Segmentations"
)

plt.tight_layout()

plt.savefig(
    "Gold_MVT_k_Sensitivity.png",
    dpi=300
)

plt.show()


# ============================================================
# 37. MVT p-VALUE SENSITIVITY PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.semilogy(
    mvt_sensitivity_df["k"],
    mvt_sensitivity_df["p_value"],
    marker="o"
)

plt.axhline(
    ALPHA,
    linestyle="--",
    label="alpha = 0.05"
)

plt.xlabel(
    "Number of Segments (k)"
)

plt.ylabel(
    "MVT p-value (log scale)"
)

plt.title(
    "MVT p-value Across Different Segmentations"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Gold_MVT_pvalue_Sensitivity.png",
    dpi=300
)

plt.show()


# ============================================================
# 38. SAVE OUTPUT FILES
# ============================================================

mvt_sensitivity_df.to_csv(
    "Gold_MVT_k_Sensitivity.csv",
    index=False
)

kstar_df.to_csv(
    "Gold_MVT_kstar_Rule.csv",
    index=False
)

segment_table.to_csv(
    "Gold_MVT_Segment_Variances.csv",
    index=False
)

comparison_table.to_csv(
    "Gold_Variance_Test_Comparison.csv",
    index=False
)

arima_table.to_csv(
    "Gold_ARIMA_Model_Selection.csv",
    index=False
)

df.to_csv(
    "Gold_Cleaned_Data.csv",
    index=False
)


# ============================================================
# 39. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL SUMMARY")
print("=" * 75)

print(
    f"Original observations : "
    f"{before_cleaning}"
)

print(
    f"Final observations    : "
    f"{len(df)}"
)

print(
    f"Log returns           : "
    f"{len(returns)}"
)

print(
    f"Selected ARIMA        : "
    f"ARIMA{best_order}"
)

print(
    f"c used                : "
    f"{c_main}"
)

print(
    f"Target k*             : "
    f"{target_k}"
)

print(
    f"Selected main k       : "
    f"{main_k}"
)

print(
    f"Segment size m        : "
    f"{main_result['m']}"
)

print(
    f"MVT statistic         : "
    f"{main_result['Q_MVT']:.10f}"
)

print(
    f"MVT df                : "
    f"{main_result['df']}"
)

print(
    f"MVT p-value           : "
    f"{main_result['p_value']:.10e}"
)

print(
    f"MVT critical value    : "
    f"{main_result['critical_value']:.10f}"
)

print(
    f"Decision              : "
    f"{'Reject H0' if main_result['reject_H0'] else 'Fail to reject H0'}"
)


# ============================================================
# 40. OUTPUT FILE LIST
# ============================================================

print("\nOutput files saved:")

print(
    "1. Gold_Cleaned_Data.csv"
)

print(
    "2. Gold_MVT_k_Sensitivity.csv"
)

print(
    "3. Gold_MVT_kstar_Rule.csv"
)

print(
    "4. Gold_MVT_Segment_Variances.csv"
)

print(
    "5. Gold_Variance_Test_Comparison.csv"
)

print(
    "6. Gold_ARIMA_Model_Selection.csv"
)

print(
    "7. Gold_Price_Series.png"
)

print(
    "8. Gold_Log_Returns.png"
)

print(
    "9. Gold_MVT_k_Sensitivity.png"
)

print(
    "10. Gold_MVT_pvalue_Sensitivity.png"
)

print("\nAnalysis completed successfully.")