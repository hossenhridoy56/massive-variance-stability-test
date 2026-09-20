import numpy as np
import pandas as pd
from scipy.stats import chi2, t, laplace, skewnorm
import time


# =========================
# Settings
# =========================

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

rng = np.random.default_rng(SEED)


# =========================
# MVT Function
# =========================

def mvt_test(x, k, alpha=0.05):

    n = len(x)
    m = n // k

    x = x[:k * m]
    segments = x.reshape(k, m)

    v = np.var(segments, axis=1, ddof=1)

    v_bar = np.mean(v)

    sigma_hat_sq = np.sum((m - 1) * v) / (n - k)

    Q_MVT = (
        (m - 1)
        / (2 * sigma_hat_sq ** 2)
        * np.sum((v - v_bar) ** 2)
    )

    p_value = chi2.sf(Q_MVT, k - 1)

    return p_value


# =========================
# Model 1
# Homoskedastic Gaussian
# =========================

type1_results = []

start_time = time.perf_counter()

for n in SAMPLE_SIZES:

    for k in K_VALUES[n]:

        reject = 0

        for r in range(R):

            x = rng.normal(0, 1, n)

            p = mvt_test(x, k, ALPHA)

            if p < ALPHA:
                reject += 1

        type1_error = reject / R
        m = n // k

        type1_results.append([
            n, k, m, type1_error
        ])

type1_time = time.perf_counter() - start_time

type1_df = pd.DataFrame(
    type1_results,
    columns=["n", "k", "m", "Type_I_Error"]
)

print("\nModel 1: Type I Error")
print(type1_df.to_string(index=False))


# =========================
# Model 2
# Single Variance Break
# =========================

SIGMA2_VALUES = [1.10, 1.25, 1.50, 2.00, 3.00, 5.00]
ETA_VALUES = [0.25, 0.50, 0.75]

power_results = []

start_time = time.perf_counter()

for n in SAMPLE_SIZES:

    for k in K_VALUES[n]:

        for sigma2_sq in SIGMA2_VALUES:

            for eta in ETA_VALUES:

                reject = 0

                b = round(eta * n)

                for r in range(R):

                    x1 = rng.normal(
                        0,
                        1,
                        b
                    )

                    x2 = rng.normal(
                        0,
                        np.sqrt(sigma2_sq),
                        n - b
                    )

                    x = np.concatenate([x1, x2])

                    p = mvt_test(x, k, ALPHA)

                    if p < ALPHA:
                        reject += 1

                power = reject / R

                power_results.append([
                    n,
                    k,
                    n // k,
                    sigma2_sq,
                    eta,
                    power
                ])

model2_time = time.perf_counter() - start_time

power_df = pd.DataFrame(
    power_results,
    columns=[
        "n",
        "k",
        "m",
        "sigma2_sq",
        "eta",
        "Power"
    ]
)

print("\nModel 2: Single Variance Break")
print(power_df.to_string(index=False))


# =========================
# Model 3
# Multiple Variance Breaks
# Variance: 1 -> 2 -> 1 -> 3
# =========================

model3_results = []

start_time = time.perf_counter()

for n in SAMPLE_SIZES:

    for k in K_VALUES[n]:

        reject = 0

        for r in range(R):

            b1 = round(0.25 * n)
            b2 = round(0.50 * n)
            b3 = round(0.75 * n)

            x1 = rng.normal(0, 1, b1)
            x2 = rng.normal(0, np.sqrt(2), b2 - b1)
            x3 = rng.normal(0, 1, b3 - b2)
            x4 = rng.normal(0, np.sqrt(3), n - b3)

            x = np.concatenate([x1, x2, x3, x4])

            p = mvt_test(x, k, ALPHA)

            if p < ALPHA:
                reject += 1

        rejection_rate = reject / R

        model3_results.append([
            n,
            k,
            n // k,
            rejection_rate
        ])

model3_time = time.perf_counter() - start_time

model3_df = pd.DataFrame(
    model3_results,
    columns=[
        "n",
        "k",
        "m",
        "Rejection_Rate"
    ]
)

print("\nModel 3: Multiple Variance Breaks")
print(model3_df.to_string(index=False))


# =========================
# Model 4
# Non-Gaussian Noise
# =========================

NON_GAUSSIAN = ["t5", "Laplace", "SkewNormal"]

model4_results = []

start_time = time.perf_counter()

for distribution in NON_GAUSSIAN:

    for n in SAMPLE_SIZES:

        for k in K_VALUES[n]:

            reject = 0

            for r in range(R):

                if distribution == "t5":

                    x = t.rvs(
                        df=5,
                        size=n,
                        random_state=rng
                    )

                    x = x / np.sqrt(5 / 3)

                elif distribution == "Laplace":

                    x = laplace.rvs(
                        loc=0,
                        scale=1 / np.sqrt(2),
                        size=n,
                        random_state=rng
                    )

                elif distribution == "SkewNormal":

                    x = skewnorm.rvs(
                        a=5,
                        size=n,
                        random_state=rng
                    )

                    x = (
                        x - np.mean(x)
                    ) / np.std(x)

                p = mvt_test(x, k, ALPHA)

                if p < ALPHA:
                    reject += 1

            rejection_rate = reject / R

            model4_results.append([
                distribution,
                n,
                k,
                n // k,
                rejection_rate
            ])

model4_time = time.perf_counter() - start_time

model4_df = pd.DataFrame(
    model4_results,
    columns=[
        "Distribution",
        "n",
        "k",
        "m",
        "Type_I_Error"
    ]
)

print("\nModel 4: Non-Gaussian Noise")
print(model4_df.to_string(index=False))


# =========================
# Model 5
# ARCH(1)
# =========================

model5_results = []

start_time = time.perf_counter()

for n in SAMPLE_SIZES:

    for k in K_VALUES[n]:

        reject = 0

        for r in range(R):

            x = np.zeros(n)
            sigma_sq = np.zeros(n)

            for t_idx in range(n):

                if t_idx == 0:
                    sigma_sq[t_idx] = 0.1
                else:
                    sigma_sq[t_idx] = (
                        0.1
                        + 0.8 * x[t_idx - 1] ** 2
                    )

                x[t_idx] = (
                    np.sqrt(sigma_sq[t_idx])
                    * rng.normal()
                )

            p = mvt_test(x, k, ALPHA)

            if p < ALPHA:
                reject += 1

        rejection_rate = reject / R

        model5_results.append([
            n,
            k,
            n // k,
            rejection_rate
        ])

model5_time = time.perf_counter() - start_time

model5_df = pd.DataFrame(
    model5_results,
    columns=[
        "n",
        "k",
        "m",
        "Rejection_Rate"
    ]
)

print("\nModel 5: ARCH(1)")
print(model5_df.to_string(index=False))


# =========================
# Model 6
# Massive Samples
# =========================

MASSIVE_SIZES = [10_000, 100_000, 1_000_000]

model6_results = []

for n in MASSIVE_SIZES:

    k = int(np.sqrt(n))

    x = rng.normal(0, 1, n)

    start = time.perf_counter()

    p = mvt_test(x, k, ALPHA)

    runtime = time.perf_counter() - start

    model6_results.append([
        n,
        k,
        n // k,
        runtime,
        p
    ])

model6_df = pd.DataFrame(
    model6_results,
    columns=[
        "n",
        "k",
        "m",
        "Runtime_Seconds",
        "p_value"
    ]
)

print("\nModel 6: Massive Samples")
print(model6_df.to_string(index=False))


# =========================
# Save Results
# =========================

type1_df.to_csv(
    "MVT_Model1_Type1_Error.csv",
    index=False
)

power_df.to_csv(
    "MVT_Model2_Power.csv",
    index=False
)

model3_df.to_csv(
    "MVT_Model3_Multiple_Breaks.csv",
    index=False
)

model4_df.to_csv(
    "MVT_Model4_NonGaussian.csv",
    index=False
)

model5_df.to_csv(
    "MVT_Model5_ARCH.csv",
    index=False
)

model6_df.to_csv(
    "MVT_Model6_Massive_Samples.csv",
    index=False
)


# =========================
# Runtime Summary
# =========================

runtime_df = pd.DataFrame({
    "Model": [
        "Model 1",
        "Model 2",
        "Model 3",
        "Model 4",
        "Model 5"
    ],
    "Runtime_Seconds": [
        type1_time,
        model2_time,
        model3_time,
        model4_time,
        model5_time
    ]
})

runtime_df.to_csv(
    "MVT_Simulation_Runtime.csv",
    index=False
)

print("\nSimulation completed.")
print(runtime_df.to_string(index=False))