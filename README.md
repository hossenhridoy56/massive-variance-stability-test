# Massive Variance Stability Test

Code and supplementary materials for the research paper:

> **A Novel Massive Variance Stability Test: Theory, Distributional Derivations, and Performance Under Structural Breaks**

## Overview

This repository contains the code and supporting materials for the **Massive Variance Stability Test (MVT)**.

MVT is a global test for checking whether variance remains stable over time. The method divides a time series into non-overlapping segments, calculates the variance within each segment, and measures the variation among these segment-wise variances.

The repository includes the Python code used for the simulations and empirical analysis, as well as the R code used to generate the manuscript figures.

---

## Methodology

For a time series of length \(n\), the observations are divided into \(k\) equal-sized segments:

$$
m=\frac{n}{k}.
$$

Let \(v_j\) be the sample variance of segment \(j\). Under the Gaussian homoskedastic null hypothesis,

$$
H_0:\sigma_1^2=\sigma_2^2=\cdots=\sigma_k^2=\sigma^2,
$$

the segment variance follows

$$
\frac{(m-1)v_j}{\sigma^2}\sim\chi^2_{m-1}.
$$

The proposed MVT statistic is

$$
Q_{\mathrm{MVT}}
=
\frac{m-1}{2\bar v^2}
\sum_{j=1}^{k}(v_j-\bar v)^2
$$

where

$$
\bar v=\frac{1}{k}\sum_{j=1}^{k}v_j.
$$

For fixed \(k\) and sufficiently large \(m\),

$$
Q_{\mathrm{MVT}}
\overset{d}{\longrightarrow}
\chi^2_{k-1}.
$$

The study also derives the noncentral chi-square formulation under variance changes and an analytical noncentrality expression for a single break occurring at a segment boundary.

---

## Segment Selection

The number of segments is selected using the practical rule

$$
k^*=\left\lfloor c\sqrt n\right\rfloor,
$$

subject to a minimum segment size.

For the Gold Futures application,

$$
m_{\min}=50.
$$

Equal segment sizes are used throughout the analysis. Therefore, a small number of observations may be excluded when \(n\) is not exactly divisible by \(k\).

---

## Monte Carlo Simulation

The simulation study uses:

- **10,000 replications**
- **Significance level:** 0.05
- **Random seed:** 12345

The simulations examine:

- Type I error;
- variance-change alternatives;
- different sample sizes and segment configurations;
- non-Gaussian distributions;
- ARCH-type dependence;
- large-sample computational performance.

The main sample sizes are \(n=200,500,2500,\) and \(10000\).

Variance-change scenarios use

$$
\sigma_2^2\in\{1.10,1.25,1.50,2,3,5\}
$$

with break locations

$$
\eta\in\{0.25,0.50,0.75\}.
$$

---

## Comparison Tests

MVT is compared with several established procedures:

- Bartlett's test
- Levene's test
- Goldfeld--Quandt test
- Breusch--Pagan test
- White's test
- Iterative Cumulative Sum of Squares (ICSS)

The comparisons are intended to show how the methods behave under different settings. They are not used to claim a universal performance ranking.

---

## Gold Futures Application

The empirical analysis uses **4,997 daily Gold Futures price observations**, covering **January 3, 2005 to February 24, 2026**.

The analysis includes:

1. Log-return construction
2. Stationarity tests
3. ARIMA modeling
4. Residual diagnostics
5. MVT testing
6. Sensitivity analysis for \(k\)
7. Segment-wise variance estimation
8. Comparison with classical tests

The final segmentation uses

$$
k=99,\qquad m=50,
$$

with 4,950 observations.

The final MVT result is

$$
Q_{\mathrm{MVT}}=1540.9529,
$$

with 98 degrees of freedom and an asymptotic p-value of approximately

$$
7.67\times10^{-258}.
$$

Because the residuals are not normally distributed and show heteroskedasticity, this p-value is treated as a **model-based reference**, not a distribution-free significance level.

---

## Repository Structure

```text
massive-variance-stability-test/
├── README.md
├── LICENSE
├── requirements.txt
├── code/
│   ├── MVT_vs_Classical_Tests.py
│   ├── gold_futures_analysis.py
│   └── MVT_FIGURES.R
└── data/
    ├── Gold Futures Historical Data.csv
    ├── MVT_Model1_Type1_Error.csv
    ├── MVT_Model2_Power.csv
    ├── MVT_Model3_Multiple_Breaks.csv
    ├── MVT_Model4_NonGaussian.csv
    ├── MVT_Model5_ARCH.csv
    ├── MVT_Model6_Massive_Samples.csv
    ├── MVT_Simulation_Runtime.csv
    └── README.md
