# Does Education Improve Health? IV Estimates from Quarter of Birth

An instrumental-variables study of the causal effect of schooling on health, using the **1980 US Census** (IPUMS, 1.94 million individuals). Following Angrist and Krueger (1991), quarter of birth interacted with compulsory-schooling rules serves as an instrument for years of education. The outcome is a work-limiting disability indicator.

## Identification

| | |
|---|---|
| **Outcome** | Work disability (`disabwrk`); transportation disability as an extension |
| **Endogenous regressor** | Years of schooling |
| **Instrument** | Quarter of birth: children born early in the year reach the school-leaving age with less schooling completed |
| **Controls** | Age and age squared; sex, race and state fixed effects; census person weights |

## Main results

| Estimator | Effect of one more year of schooling on P(work disability) |
|---|---|
| OLS | −0.023 (SE < 0.001) |
| 2SLS, quarter-of-birth instruments | **−0.068 (SE 0.013)** |
| First-stage F-statistic | 1,789 |

The 2SLS estimate is about three times larger than OLS and remains strongly significant; the first stage is strong. The analysis also reproduces the Angrist–Krueger cohort figures for schooling and wages by year and quarter of birth.

## Methods

- Reconstruction of birth cohorts from census age and quarter of birth
- Replication of the Angrist–Krueger Figures I–III for schooling, and the same figures for wages
- Weighted OLS, manual two-stage least squares, and `linearmodels` IV2SLS with heteroskedasticity-robust standard errors
- First-stage strength diagnostics and a control-function (Durbin–Wu–Hausman) test of endogeneity

## Repository structure

```
analysis.md   # full analysis: code, regression output and interpretation
```

## Reproducing

Download a 1980 Census extract from [IPUMS USA](https://usa.ipums.org/) with `educ`, `birthqtr`, `age`, `perwt`, `incwage`, `race`, `sex`, `statefip`, `disabwrk` and `disabtrn`, and save it as `usa_1980.dta`.

```bash
pip install pandas numpy statsmodels linearmodels matplotlib seaborn
```

## Reference

Angrist, J. D., and Krueger, A. B. (1991). Does Compulsory School Attendance Affect Schooling and Earnings? *Quarterly Journal of Economics*, 106(4), 979–1014.

## Context

Developed during the MSc in Economics at Sharif University of Technology (Applied Econometrics, Spring 2025).

---
**Shayan Pourmasoomi** · MSc Economics, Sharif University of Technology · [LinkedIn](https://www.linkedin.com/in/shayan-pourmasoomi-8b6206389)
