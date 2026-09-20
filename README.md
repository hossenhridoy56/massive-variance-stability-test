# Massive Variance Stability Test

Code and supplementary materials for the research paper:

> **A Novel Massive Variance Stability Test: Theory, Distributional Derivations, and Performance Under Structural Breaks**

## Overview

This repository contains the computational materials supporting the proposed **Massive Variance Stability Test (MVT)**.

The MVT is a global test for assessing variance stability in long time series. The method divides an observed series into non-overlapping segments of equal size, calculates the sample variance within each segment, and measures the variation among the segment-wise variance estimates.

Under the Gaussian homoskedastic framework, the segment variances have a known asymptotic distribution, which leads to a chi-square-based test statistic. The framework also provides an alternative noncentral chi-square formulation for variance changes and an analytical noncentrality expression for a single variance break occurring at a segment boundary.

The repository provides the Python implementation used for the Monte Carlo simulations and empirical calculations, together with the R code used to generate the figures presented in the manuscript.

---

## Research Objectives

The study develops and evaluates a global variance-stability testing framework with the following objectives:

1. Develop a simple test statistic based on variation among non-overlapping segment variances.
2. Derive the distribution of the segment-wise sample variances under Gaussian homoskedasticity.
3. Obtain the asymptotic chi-square distribution of the proposed MVT statistic.
4. Derive the alternative noncentral chi-square distribution.
5. Obtain an analytical expression for the noncentrality parameter under a single variance break.
6. Develop a practical rule for selecting the number of segments.
7. Evaluate Type I error control and statistical power through Monte Carlo simulation.
8. Examine the behavior of the test under non-Gaussian distributions and dependence structures.
9. Compare MVT with established variance-stability and heteroskedasticity procedures.
10. Demonstrate the method using daily Gold Futures data.

---

## Methodology

Suppose a time series of length \(n\) is divided into \(k\) non-overlapping segments of equal size

$$
m=\frac{n}{k}.
$$

Let \(v_j\) denote the sample variance calculated from segment \(j\), where \(j=1,\ldots,k\).

Under the Gaussian homoskedastic null hypothesis,

$$
H_0:\sigma_1^2=\sigma_2^2=\cdots=\sigma_k^2=\sigma^2,
$$

the segment variance satisfies

$$
\frac{(m-1)v_j}{\sigma^2}\sim\chi^2_{m-1}.
$$

Consequently,

$$
E(v_j)=\sigma^2,
$$

and

$$
\operatorname{Var}(v_j)=\frac{2\sigma^4}{m-1}.
$$

Because the segments are non-overlapping and the observations are independent under the basic framework, the segment variance estimates are independent.

For fixed \(k\) and sufficiently large \(m\),

$$
\mathbf v
\overset{a}{\sim}
N_k\left(
\sigma^2\mathbf 1_k,
\frac{2\sigma^4}{m-1}I_k
\right).
$$

Let

$$
\bar v=\frac{1}{k}\sum_{j=1}^k v_j.
$$

The proposed MVT statistic is

$$
Q_{\mathrm{MVT}}
=
\frac{m-1}{2\bar v^2}
\sum_{j=1}^{k}(v_j-\bar v)^2.
$$

Under \(H_0\), for fixed \(k\) and \(m\rightarrow\infty\),

$$
Q_{\mathrm{MVT}}
\overset{d}{\longrightarrow}
\chi^2_{k-1}.
$$

The corresponding asymptotic p-value is

$$
p
=
1-F_{\chi^2_{k-1}}
\left(Q_{\mathrm{MVT}}\right).
$$

The alternative formulation is

$$
Q_{\mathrm{MVT}}
\overset{a}{\sim}
\chi^2_{k-1}(\lambda),
$$

where the noncentrality parameter is

$$
\lambda
=
\frac{m-1}{2\sigma_0^4}
\sum_{j=1}^{k}
(\sigma_j^2-\bar\sigma^2)^2.
$$

For a single variance break aligned with a segment boundary, an analytical expression for the noncentrality parameter is derived in the manuscript.

---

## Practical Segment Selection

The number of segments \(k\) is an important practical component of the method.

The study considers the square-root rule

$$
k^*=\left\lfloor c\sqrt n\right\rfloor,
$$

where \(c\) is selected from a practical range of candidate values.

A minimum segment size is imposed to avoid excessively short segments. In the empirical Gold Futures application, the minimum segment size is

$$
m_{\min}=50.
$$

The final admissible value of \(k\) is therefore constrained by

$$
m=\frac{n}{k}\geq m_{\min}.
$$

Because the theoretical development assumes equal segment sizes, observations may be excluded when \(n\) is not exactly divisible by the selected \(k\).

---

## Monte Carlo Simulation

The finite-sample behavior of MVT is investigated using Monte Carlo simulations.

### General simulation settings

* Number of Monte Carlo replications: **10,000**
* Significance level: **0.05**
* Random seed: **12345**

The simulations examine several sample sizes and segment configurations.

### Sample sizes and segment configurations

The following values of \(n\) and \(k\) are considered:

| Sample size \(n\) | Number of segments \(k\) |
| ----------------: | ------------------------ |
|               200 | 4, 5, 10                 |
|               500 | 5, 10                    |
|             2,500 | 5, 10, 25                |
|            10,000 | 10, 20, 50               |

### Variance-change settings

Variance alternatives are examined using

$$
\sigma_2^2
\in
\{1.10,1.25,1.50,2,3,5\},
$$

with variance-change locations controlled through

$$
\eta\in\{0.25,0.50,0.75\}.
$$

### Additional data-generating scenarios

The simulation study also examines the behavior of MVT under:

* Gaussian homoskedastic data for Type I error calibration;
* variance-change alternatives;
* heavy-tailed distributions;
* skewed distributions;
* ARCH-type dependence.

The dependence scenario is intended as a stress test because serial dependence violates the independence assumption underlying the basic asymptotic calibration.

A large-sample computational experiment is also performed for

$$
n\in\{10,000,\ 100,000,\ 1,000,000\},
$$

with

$$
k=\lfloor\sqrt n\rfloor.
$$

---

## Comparison with Classical Procedures

The simulation study compares MVT with several established procedures, including:

* **Bartlett's test**
* **Levene's test**
* **Goldfeld--Quandt test**
* **Breusch--Pagan test**
* **White's test**
* **Iterative Cumulative Sum of Squares (ICSS)**

These procedures address related but not identical variance-stability or heteroskedasticity questions. Therefore, the comparisons are intended to describe differences in behavior across simulation settings rather than establish a universal performance ordering.

---

## Empirical Application: Gold Futures

The proposed method is applied to daily Gold Futures data.

The empirical analysis uses **4,997 daily price observations**, corresponding to **4,996 log returns**, covering the period from **January 3, 2005 to February 24, 2026**.

The analysis includes:

1. Construction of daily log returns.
2. Stationarity diagnostics.
3. ARIMA modeling of the return series.
4. Residual diagnostics.
5. MVT variance-stability testing.
6. Sensitivity analysis across different values of \(k\).
7. Selection of the final segmentation.
8. Segment-wise variance estimation.
9. Comparison with classical variance-stability and heteroskedasticity tests.

For the final empirical segmentation,

$$
k=99,\qquad m=50,
$$

using 4,950 observations from the 4,996 available log returns. The remaining observations are excluded because the theoretical formulation uses equal-sized segments.

The final MVT statistic is

$$
Q_{\mathrm{MVT}}=1540.9529,
$$

with

$$
df=98,
$$

and an asymptotic chi-square p-value of approximately

$$
7.67\times10^{-258}.
$$

The segment-wise variance estimates show substantial variation across the series.

Because the empirical residuals depart materially from the Gaussian assumption and exhibit heteroskedasticity, the reported chi-square p-value is interpreted as a **model-based reference rather than a distribution-free significance level**.

---

## Repository Structure

The repository is organized as follows:

```text
massive-variance-stability-test/
│
├── README.md
│
├── simulation/
│   ├── MVT_simulation.py
│   ├── classical_tests.py
│   └── simulation_results/
│
├── figures/
│   ├── Figure_1_Type_I_Error.R
│   ├── Figure_2_Power_Comparison.R
│   ├── Figure_3_Gold_Prices_Returns.R
│   ├── Figure_4_MVT_k_Sensitivity.R
│   └── Figure_5_Segment_Variances.R
│
├── empirical/
│   └── gold_futures_analysis.py
│
├── data/
│   └── README.md
│
├── supplementary/
│   └── simulation_results.xlsx
│
├── CITATION.cff
└── LICENSE
```

The exact filenames may differ slightly from the final repository organization.

---

## Programming Languages

The computational work uses two programming environments:

### Python

Python is used for:

* Monte Carlo simulations;
* implementation of the MVT;
* implementation of the classical comparison tests;
* ICSS calculations;
* empirical statistical calculations;
* numerical analysis.

### R

R is used for:

* generation of the manuscript figures;
* visualization of simulation results;
* visualization of the empirical results.

---

## Reproducibility

To reproduce the simulation results:

1. Clone or download this repository.
2. Install the required Python packages.
3. Set the random seed to `12345`.
4. Run the simulation scripts.
5. Save the generated simulation results.
6. Run the R figure-generation scripts using the corresponding simulation-result files.
7. Compare the generated tables and figures with those reported in the manuscript.

The simulation study uses 10,000 Monte Carlo replications for each simulation setting.

For computationally intensive settings, particularly the larger simulation scenarios, execution time may vary depending on processor, memory, Python version, and installed package versions.

---

## Statistical Software

The main computational environments are:

* Python for simulation and statistical computation;
* R for figure generation.

Exact package versions and computational-environment details will be documented in the repository as the reproducibility materials are finalized.

---

## Data Availability

The empirical analysis uses daily Gold Futures data.

The repository will provide the data-source information and instructions required to reproduce the empirical analysis. Raw data will only be redistributed when permitted by the applicable data-source terms.

The simulation data are generated programmatically and can be reproduced using the provided Python code.

---

## Supporting Information

The supplementary materials associated with the manuscript include:

* Complete Python simulation code;
* R figure-generation code;
* Complete simulation results;
* Additional computational details.

These materials are intended to facilitate reproduction of the reported simulation results and empirical figures.

---

## Citation

If you use the MVT methodology, code, or simulation materials in academic work, please cite the associated paper:

> Md. Matiur Rahman Molla, Md. Hridoy Hossen, Md. Murad Hossain, Md. Raihan Hossen, and Md. Mahabubur Rahman. *A Novel Massive Variance Stability Test: Theory, Distributional Derivations, and Performance Under Structural Breaks.*

A formal citation entry will be added after publication.

---

## Authors

**Md. Matiur Rahman Molla**
Department of Statistics and Data Science
Islamic University
Kushtia-7000, Bangladesh

**Md. Hridoy Hossen**
Department of Statistics and Data Science
Islamic University
Kushtia-7000, Bangladesh
ORCID: [0009-0002-5573-4688](https://orcid.org/0009-0002-5573-4688)

**Md. Murad Hossain**
Gopalganj Science and Technology University
Bangladesh

**Md. Raihan Hossen**
Department of Statistics and Data Science
Islamic University
Kushtia-7000, Bangladesh

**Md. Mahabubur Rahman**
Department of Statistics and Data Science
Islamic University
Kushtia-7000, Bangladesh

---

## License

The code in this repository is distributed under the license specified in the `LICENSE` file.

The manuscript and associated research materials remain subject to their respective copyright and publication conditions.

---

## Repository

Source code and supplementary computational materials:

https://github.com/hossenhridoy56/massive-variance-stability-test

---

## Status

This repository is under active development while the manuscript and supplementary materials are being finalized.

The repository contents may be updated to reflect revisions to the manuscript, simulation code, figures, and supplementary materials.
