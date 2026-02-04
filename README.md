# Instrumental Variables (IV) — Quarter of Birth, Education, and Health (US Census 1980)

This repository implements an **Instrumental Variables / 2SLS** analysis to estimate the causal effect of **education on health outcomes** using the **1980 US Census** data.

The project follows the classic approach of **Angrist & Krueger (1991)**, using **Quarter of Birth (QOB)** as an instrument for education, motivated by compulsory schooling laws. It also includes replication-style visual checks (AK91 Figures I–III) and standard IV diagnostics.

> Course context: Applied Metrics (MSc) — Problem Set 09 (Spring 2025): Instrumental Variables.

---

## Research Question

**Does education causally improve health?**

We proxy health using disability indicators available in the 1980 Census:
- `disabwrk` — Work disability
- `disabtrn` — Transportation disability (optional extension)

Because education may be endogenous (e.g., omitted ability, family background, measurement error), we use an IV strategy:
- **Instrument**: Quarter of Birth (QOB)
- **Endogenous variable**: Years of education (`educ`)
- **Outcome**: Disability measure (`disabwrk`)

---

## Data

Expected input file:
- `usa_1980.dta`

Key variables:
- `educ` — years of schooling (endogenous regressor)
- `birthqtr` — quarter of birth (instrument source)
- `age` — age (used to infer birth year)
- `perwt` — person weight (used for weighted estimation)
- `incwage` — wage income (used for wage-figure replications)
- `race`, `sex`, `statefip` — controls / fixed effects
- `disabwrk`, `disabtrn` — disability outcomes

---

## What the Code Does

### 1) Reconstruct cohort timing and replicate AK-style figures
Since the IPUMS extract does not directly provide date of birth, we compute:
- `birthyr = 1980 - age`

Then we replicate education patterns by:
- Year of birth × quarter of birth (Figure I style)
- Q1–Q4 differences over cohorts (Figure II style)
- Education by age and QOB (Figure III style)

We also repeat the same plots using **wages** (`incwage`) instead of education.

Outputs:
- `figure1.png`, `figure2.png`, `figure3.png`
- `figure1_wages.png`, `figure2_wages.png`, `figure3_wages.png`

---

## Econometric Models

### OLS (baseline)
We estimate a weighted OLS model:

Healthᵢ = β₀ + β₁ educᵢ + f(ageᵢ) + controlsᵢ + εᵢ

with:
- age and age²
- race, sex, and state fixed effects (dummies)
- weights: `perwt`
- robust (heteroskedasticity-robust) SEs

### 2SLS / IV (main specification)
**First stage:**
educᵢ = π₀ + π₁ QOBᵢ + f(ageᵢ) + controlsᵢ + νᵢ

**Second stage:**
Healthᵢ = β₀ + β₁ educ_hatᵢ + f(ageᵢ) + controlsᵢ + uᵢ

We estimate IV both:
- manually (two-step WLS)
- using `linearmodels.iv.IV2SLS`

---

## Endogeneity Test (Durbin–Wu–Hausman style)

We test whether education is endogenous by including the first-stage residual in the structural equation:

Healthᵢ = β₀ + β₁ educᵢ + γ residᵢ + controlsᵢ + uᵢ

If `γ` is statistically significant, it suggests endogeneity of education (given a valid instrument).

---

## Installation

```bash
pip install pandas numpy matplotlib seaborn statsmodels linearmodels
