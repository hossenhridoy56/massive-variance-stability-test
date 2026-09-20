"""
MVT vs Classical Variance-Stability Tests
==========================================

Monte Carlo comparison for:

    1. MVT
    2. Bartlett
    3. Levene
    4. Goldfeld-Quandt (GQ)
    5. Breusch-Pagan (BP)
    6. White
    7. Inclan-Tiao ICSS

Models:
    Model 1: Homoskedastic Gaussian noise
             -> Type I error

    Model 2: Single variance break
             -> Power

All tests use R = 10,000 replications per configuration.

Important conventions:
    - MVT, Bartlett and Levene use the SAME segmentation k.
    - GQ, BP, White and ICSS do not depend on k.
    - GQ: equal half-sample split, drop=0, two-sided.
    - BP: auxiliary regression on [1, t].
    - White: auxiliary regression on [1, t, t^2].
    - ICSS: Inclan-Tiao global variance-stability statistic.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist
import time


# ============================================================
# 1. SETTINGS
# ============================================================

R = 10000
ALPHA = 0.05
SEED = 12345

SAMPLE_SIZES = [200, 500, 2500, 10000]

K_VALUES = {
    200: [4, 5, 10],
    500: [5, 10],
    2500: [5, 10, 25],
    10000: [10, 20, 50]
}

SIGMA2_VALUES = [1.10, 1.25, 1.50, 2.00, 3.00, 5.00]
ETA_VALUES = [0.25, 0.50, 0.75]

# Process 5,000 replications at a time
# to control RAM usage.
BATCH = 5000

rng = np.random.default_rng(SEED)


# ============================================================
# 2. MVT
# ============================================================

def mvt_reject(X, k, alpha=ALPHA):

    Rb, n = X.shape

    m = n // k
    usable = k * m

    # Divide observations into k non-overlapping segments
    seg = X[:, :usable].reshape(Rb, k, m)

    # Segment variances
    v = seg.var(axis=2, ddof=1)

    # Mean segment variance
    v_bar = v.mean(axis=1, keepdims=True)

    # Unbiased pooled variance estimator
    sigma_hat_sq = (
        ((m - 1) * v).sum(axis=1) / (n - k)
    )

    # MVT statistic
    Q = (
        (m - 1)
        / (2 * sigma_hat_sq ** 2)
        * ((v - v_bar) ** 2).sum(axis=1)
    )

    # Chi-square approximation
    p = chi2.sf(Q, df=k - 1)

    return p < alpha


# ============================================================
# 3. BARTLETT TEST
# ============================================================

def bartlett_reject(X, k, alpha=ALPHA):

    Rb, n = X.shape

    m = n // k
    usable = k * m

    seg = X[:, :usable].reshape(Rb, k, m)

    # Group variances
    s2 = seg.var(axis=2, ddof=1)

    N = k * m

    # Pooled variance
    pooled = (
        ((m - 1) * s2).sum(axis=1)
        / (N - k)
    )

    # Bartlett statistic
    log_s2 = np.log(
        np.clip(s2, 1e-300, None)
    )

    stat = (
        (N - k) * np.log(
            np.clip(pooled, 1e-300, None)
        )
        - (m - 1) * log_s2.sum(axis=1)
    )

    # Bartlett correction
    C = (
        1
        + (k + 1)
        / (3 * k * (m - 1))
    )

    chi2_stat = stat / C

    p = chi2.sf(
        chi2_stat,
        df=k - 1
    )

    return p < alpha


# ============================================================
# 4. LEVENE TEST
# ============================================================

def levene_reject(X, k, alpha=ALPHA):

    Rb, n = X.shape

    m = n // k
    usable = k * m

    seg = X[:, :usable].reshape(Rb, k, m)

    # --------------------------------------------------------
    # Levene's original mean-based version:
    #
    # z_ij = |x_ij - group_mean_j|
    # --------------------------------------------------------

    group_mean = seg.mean(axis=2, keepdims=True)

    z = np.abs(seg - group_mean)

    # Mean absolute deviation within each group
    z_bar_j = z.mean(axis=2)

    # Overall mean of z
    z_bar = z.mean(axis=(1, 2))

    # Between-group sum of squares
    ss_between = (
        m
        * ((z_bar_j - z_bar[:, None]) ** 2).sum(axis=1)
    )

    # Within-group sum of squares
    ss_within = (
        (z - z_bar_j[:, :, None]) ** 2
    ).sum(axis=(1, 2))

    # Levene F statistic
    df1 = k - 1
    df2 = k * (m - 1)

    F = (
        (ss_between / df1)
        / (ss_within / df2)
    )

    p = f_dist.sf(
        F,
        df1,
        df2
    )

    return p < alpha


# ============================================================
# 5. OLS RESIDUALS FOR GQ / BP / WHITE
# ============================================================

def ols_residuals(X, design):

    # X shape:
    #       (Rbatch, n)
    #
    # design shape:
    #       (n, p)

    P = np.linalg.pinv(design)

    beta = X @ P.T

    fitted = beta @ design.T

    residuals = X - fitted

    return residuals


# ============================================================
# 6. BP + WHITE
# ============================================================

def bp_white_reject(X, alpha=ALPHA):

    Rb, n = X.shape

    t = np.arange(
        n,
        dtype=float
    )

    # --------------------------------------------------------
    # Original regression:
    #
    # X_t = beta_0 + beta_1 t + error_t
    # --------------------------------------------------------

    design = np.column_stack([
        np.ones(n),
        t
    ])

    P = np.linalg.pinv(design)

    beta = X @ P.T

    fitted = beta @ design.T

    resid = X - fitted

    e2 = resid ** 2

    e2_mean = e2.mean(
        axis=1,
        keepdims=True
    )

    sst = (
        (e2 - e2_mean) ** 2
    ).sum(axis=1)

    # ========================================================
    # Breusch-Pagan
    #
    # e_t^2 = alpha_0 + alpha_1 t + u_t
    # ========================================================

    beta_bp = e2 @ P.T

    fitted_bp = beta_bp @ design.T

    ssr_bp = (
        (fitted_bp - e2_mean) ** 2
    ).sum(axis=1)

    r2_bp = np.divide(
        ssr_bp,
        sst,
        out=np.zeros_like(ssr_bp),
        where=sst > 0
    )

    LM_bp = n * r2_bp

    p_bp = chi2.sf(
        LM_bp,
        df=1
    )

    # ========================================================
    # White
    #
    # e_t^2 = alpha_0 + alpha_1 t
    #                 + alpha_2 t^2 + u_t
    # ========================================================

    design_w = np.column_stack([
        np.ones(n),
        t,
        t ** 2
    ])

    Pw = np.linalg.pinv(design_w)

    beta_w = e2 @ Pw.T

    fitted_w = beta_w @ design_w.T

    ssr_w = (
        (fitted_w - e2_mean) ** 2
    ).sum(axis=1)

    r2_w = np.divide(
        ssr_w,
        sst,
        out=np.zeros_like(ssr_w),
        where=sst > 0
    )

    LM_white = n * r2_w

    p_white = chi2.sf(
        LM_white,
        df=2
    )

    reject_bp = p_bp < alpha
    reject_white = p_white < alpha

    return (
        reject_bp,
        reject_white,
        resid
    )


# ============================================================
# 7. GOLDfeld-QUANDT TEST
# ============================================================

def gq_reject(resid, alpha=ALPHA):

    Rb, n = resid.shape

    # Equal half-sample split
    # No observations dropped
    half = n // 2

    r1 = resid[:, :half]

    r2 = resid[:, n - half:]

    n1 = r1.shape[1]
    n2 = r2.shape[1]

    # Regression has 2 parameters:
    # intercept + time
    df1 = n1 - 2
    df2 = n2 - 2

    var1 = (
        (r1 ** 2).sum(axis=1)
        / df1
    )

    var2 = (
        (r2 ** 2).sum(axis=1)
        / df2
    )

    Fstat = np.divide(
        var2,
        var1,
        out=np.full_like(
            var1,
            np.nan
        ),
        where=var1 > 0
    )

    # Two-sided GQ test:
    # make F >= 1
    F_use = np.where(
        Fstat >= 1,
        Fstat,
        1 / Fstat
    )

    p = f_dist.sf(
        F_use,
        df2,
        df1
    )

    # Two-sided p-value
    p = np.minimum(
        1.0,
        2 * p
    )

    return p < alpha


# ============================================================
# 8. INCLAN-TIAO ICSS
# ============================================================

def icss_reject(X, alpha=ALPHA):

    Rb, n = X.shape

    # Cumulative sum of squared observations
    csum = np.cumsum(
        X ** 2,
        axis=1
    )

    # Total sum of squares
    CT = csum[:, -1:]

    # D_k = C_k / C_T - k/T
    k_index = np.arange(
        1,
        n,
        dtype=float
    )

    D = (
        csum[:, :-1] / CT
        - k_index[None, :] / n
    )

    # Inclan-Tiao global statistic
    IT = (
        np.sqrt(n / 2.0)
        * np.max(
            np.abs(D),
            axis=1
        )
    )

    # 5% asymptotic critical value
    critical = 1.358

    return IT > critical


# ============================================================
# 9. BATCHED SIMULATION RUNNER
# ============================================================

def run_batches(
    n,
    k,
    R,
    gen_fn,
    alpha=ALPHA,
    batch=BATCH
):

    counts = {
        "MVT": 0,
        "Bartlett": 0,
        "Levene": 0,
        "GQ": 0,
        "White": 0,
        "BP": 0,
        "ICSS": 0
    }

    done = 0

    while done < R:

        Rb = min(
            batch,
            R - done
        )

        # Generate simulated data
        X = gen_fn(Rb)

        # ----------------------------------------------------
        # MVT
        # ----------------------------------------------------

        counts["MVT"] += (
            mvt_reject(
                X,
                k,
                alpha
            ).sum()
        )

        # ----------------------------------------------------
        # Bartlett
        # ----------------------------------------------------

        counts["Bartlett"] += (
            bartlett_reject(
                X,
                k,
                alpha
            ).sum()
        )

        # ----------------------------------------------------
        # Levene
        # ----------------------------------------------------

        counts["Levene"] += (
            levene_reject(
                X,
                k,
                alpha
            ).sum()
        )

        # ----------------------------------------------------
        # BP + White + residuals
        # ----------------------------------------------------

        (
            rej_bp,
            rej_white,
            resid
        ) = bp_white_reject(
            X,
            alpha
        )

        counts["BP"] += rej_bp.sum()

        counts["White"] += rej_white.sum()

        # ----------------------------------------------------
        # GQ
        # ----------------------------------------------------

        counts["GQ"] += (
            gq_reject(
                resid,
                alpha
            ).sum()
        )

        # ----------------------------------------------------
        # ICSS
        # ----------------------------------------------------

        counts["ICSS"] += (
            icss_reject(
                X,
                alpha
            ).sum()
        )

        done += Rb

        # Free memory
        del X
        del resid
        del rej_bp
        del rej_white

    # Convert counts to rejection rates
    return {
        name: count / R
        for name, count in counts.items()
    }


# ============================================================
# 10. MODEL 1
#     HOMOSKEDASTIC GAUSSIAN NOISE
# ============================================================

def gen_model1(Rb, n):

    return rng.standard_normal(
        (Rb, n)
    )


print("\n==========================================")
print("MODEL 1: TYPE I ERROR")
print("==========================================\n")

model1_rows = []

t0 = time.perf_counter()

for n in SAMPLE_SIZES:

    for k in K_VALUES[n]:

        m = n // k

        rates = run_batches(
            n=n,
            k=k,
            R=R,
            gen_fn=lambda Rb, n=n:
                gen_model1(Rb, n)
        )

        row = {
            "n": n,
            "k": k,
            "m": m,
            "R": R,
            "alpha": ALPHA,
            **rates
        }

        model1_rows.append(row)

        elapsed = time.perf_counter() - t0

        print(
            f"[Model 1] "
            f"n={n:<5} "
            f"k={k:<3} "
            f"m={m:<5} "
            f"completed | "
            f"{elapsed:.1f}s"
        )


model1_df = pd.DataFrame(
    model1_rows
)

model1_df.to_csv(
    "MVT_vs_Classical_Model1_TypeI.csv",
    index=False
)

print("\nModel 1 results:\n")

print(
    model1_df.to_string(
        index=False
    )
)


# ============================================================
# 11. MODEL 2
#     SINGLE VARIANCE BREAK
# ============================================================

def gen_model2(
    Rb,
    n,
    sigma2_sq,
    eta
):

    b = round(
        eta * n
    )

    X = np.empty(
        (Rb, n)
    )

    # Before break
    X[:, :b] = rng.standard_normal(
        (Rb, b)
    )

    # After break
    X[:, b:] = (
        rng.standard_normal(
            (Rb, n - b)
        )
        * np.sqrt(sigma2_sq)
    )

    return X


print("\n==========================================")
print("MODEL 2: POWER")
print("==========================================\n")

model2_rows = []

t0 = time.perf_counter()

for n in SAMPLE_SIZES:

    for k in K_VALUES[n]:

        m = n // k

        for sigma2_sq in SIGMA2_VALUES:

            for eta in ETA_VALUES:

                rates = run_batches(
                    n=n,
                    k=k,
                    R=R,
                    gen_fn=lambda Rb,
                               n=n,
                               s=sigma2_sq,
                               e=eta:

                        gen_model2(
                            Rb,
                            n,
                            s,
                            e
                        )
                )

                row = {
                    "n": n,
                    "k": k,
                    "m": m,
                    "sigma2_sq": sigma2_sq,
                    "eta": eta,
                    "R": R,
                    "alpha": ALPHA,
                    **rates
                }

                model2_rows.append(
                    row
                )

        elapsed = time.perf_counter() - t0

        print(
            f"[Model 2] "
            f"n={n:<5} "
            f"k={k:<3} "
            f"m={m:<5} "
            f"completed | "
            f"{elapsed:.1f}s"
        )


model2_df = pd.DataFrame(
    model2_rows
)

model2_df.to_csv(
    "MVT_vs_Classical_Model2_Power.csv",
    index=False
)


print("\nModel 2 results: first 20 rows\n")

print(
    model2_df.head(20).to_string(
        index=False
    )
)


print("\n==========================================")
print("SIMULATION COMPLETED")
print("==========================================")

print(
    "\nFiles created:"
)

print(
    "1. MVT_vs_Classical_Model1_TypeI.csv"
)

print(
    "2. MVT_vs_Classical_Model2_Power.csv"
)