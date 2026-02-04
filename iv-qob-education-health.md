```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
```


```python
data = pd.read_stata("usa_1980.dta", convert_categoricals=False)
```


```python
data.columns
```




    Index(['year', 'datanum', 'serial', 'hhwt', 'region', 'statefip', 'metro',
           'gq', 'pernum', 'perwt', 'sex', 'age', 'birthqtr', 'marst', 'marrno',
           'agemarr', 'race', 'raced', 'hispan', 'hispand', 'bpl', 'bpld',
           'higrade', 'higraded', 'educ', 'educd', 'empstat', 'empstatd',
           'labforce', 'occ', 'ind', 'wkswork1', 'hrswork1', 'wksunemp',
           'yrlastwk', 'workedyr', 'inctot', 'incwage', 'sei', 'migrate5',
           'migrate5d', 'work5yr', 'disabwrk', 'disabtrn', 'vetstat', 'vetstatd'],
          dtype='object')




```python
# Step 5: OLS and 2SLS Regressions (Question 4)
# Sample selection: Adults aged 25-64, non-missing education and disability
data_reg = data[(data['age'] >= 25) & (data['age'] <= 64) & 
                data['educ'].notna() & data['disabwrk'].notna()].copy()

# Add age squared and dummy variables for controls
data_reg['age_sq'] = data_reg['age'] ** 2
data_reg = pd.get_dummies(data_reg, columns=['sex', 'race', 'statefip'], drop_first=True)

# OLS Regression (weighted by perwt)
ols_formula = 'disabwrk ~ educ + age + age_sq + ' + ' + '.join([col for col in data_reg.columns if col.startswith('sex_') or col.startswith('race_') or col.startswith('statefip_')])
ols_model = smf.wls(ols_formula, data=data_reg, weights=data_reg['perwt']).fit(cov_type='HC1')
print("OLS Results:")
print(ols_model.summary())

# 2SLS Regression
# First stage (weighted)
first_stage_formula = 'educ ~ qob1 + qob2 + qob3 + age + age_sq + ' + ' + '.join([col for col in data_reg.columns if col.startswith('sex_') or col.startswith('race_') or col.startswith('statefip_')])
first_stage = smf.wls(first_stage_formula, data=data_reg, weights=data_reg['perwt']).fit()
data_reg['educ_hat'] = first_stage.fittedvalues

# Second stage (weighted)
second_stage_formula = 'disabwrk ~ educ_hat + age + age_sq + ' + ' + '.join([col for col in data_reg.columns if col.startswith('sex_') or col.startswith('race_') or col.startswith('statefip_')])
second_stage = smf.wls(second_stage_formula, data=data_reg, weights=data_reg['perwt']).fit(cov_type='HC1')
print("2SLS Second Stage Results (Manual):")
print(second_stage.summary())

# 2SLS using IV2SLS (weighted)
iv_formula = 'disabwrk ~ 1 + age + age_sq + ' + ' + '.join([col for col in data_reg.columns if col.startswith('sex_') or col.startswith('race_') or col.startswith('statefip_')]) + ' + [educ ~ qob1 + qob2 + qob3]'
iv_model = IV2SLS.from_formula(iv_formula, data=data_reg, weights=data_reg['perwt']).fit(cov_type='robust')
print("2SLS Results (IV2SLS):")
print(iv_model.summary)

# Step 6: Hausman Test for Endogeneity (Question 4c)
data_reg['resid'] = first_stage.resid
hausman_formula = 'disabwrk ~ educ + resid + age + age_sq + ' + ' + '.join([col for col in data_reg.columns if col.startswith('sex_') or col.startswith('race_') or col.startswith('statefip_')])
hausman_model = smf.wls(hausman_formula, data=data_reg, weights=data_reg['perwt']).fit(cov_type='HC1')
print("Hausman Test for Endogeneity:")
print(hausman_model.summary().tables[1])

# Step 7: First Stage Strength (Question 4d)
f_stat = first_stage.fvalue
print("First Stage F-Statistic:", f_stat)
```

    Average Education by Quarter of Birth:
    birthqtr
    1    6.424040
    2    6.479475
    3    6.506075
    4    6.500461
    Name: educ, dtype: float64
    OLS Results:
    

    C:\Users\SHAYAN\anaconda3\Lib\site-packages\numpy\core\fromnumeric.py:88: RuntimeWarning: overflow encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    

                                WLS Regression Results                            
    ==============================================================================
    Dep. Variable:               disabwrk   R-squared:                       0.036
    Model:                            WLS   Adj. R-squared:                  0.036
    Method:                 Least Squares   F-statistic:                     748.7
    Date:                Thu, 05 Jun 2025   Prob (F-statistic):               0.00
    Time:                        10:48:21   Log-Likelihood:                    inf
    No. Observations:             1938040   AIC:                              -inf
    Df Residuals:                 1937980   BIC:                              -inf
    Df Model:                          59                                         
    Covariance Type:                  HC1                                         
    =======================================================================================
                              coef    std err          z      P>|z|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept               1.0836      0.003    371.402      0.000       1.078       1.089
    race_2[T.True]          0.0605      0.001     55.297      0.000       0.058       0.063
    race_3[T.True]          0.0610      0.004     13.838      0.000       0.052       0.070
    race_4[T.True]         -0.0341      0.003    -13.042      0.000      -0.039      -0.029
    race_5[T.True]         -0.0313      0.003    -10.064      0.000      -0.037      -0.025
    race_6[T.True]         -0.0275      0.002    -16.238      0.000      -0.031      -0.024
    race_7[T.True]         -0.0081      0.005     -1.559      0.119      -0.018       0.002
    statefip_2[T.True]     -0.0263      0.005     -5.008      0.000      -0.037      -0.016
    statefip_4[T.True]     -0.0031      0.003     -0.908      0.364      -0.010       0.004
    statefip_5[T.True]      0.0294      0.004      7.138      0.000       0.021       0.037
    statefip_6[T.True]      0.0061      0.002      2.429      0.015       0.001       0.011
    statefip_8[T.True]     -0.0020      0.003     -0.662      0.508      -0.008       0.004
    statefip_9[T.True]     -0.0132      0.003     -4.257      0.000      -0.019      -0.007
    statefip_10[T.True]    -0.0076      0.006     -1.360      0.174      -0.019       0.003
    statefip_11[T.True]     0.0064      0.006      1.086      0.277      -0.005       0.018
    statefip_12[T.True]    -0.0074      0.003     -2.727      0.006      -0.013      -0.002
    statefip_13[T.True]     0.0075      0.003      2.473      0.013       0.002       0.013
    statefip_15[T.True]     0.0099      0.004      2.300      0.021       0.001       0.018
    statefip_16[T.True]    -0.0022      0.005     -0.475      0.635      -0.011       0.007
    statefip_17[T.True]    -0.0096      0.003     -3.664      0.000      -0.015      -0.004
    statefip_18[T.True]    -0.0114      0.003     -3.950      0.000      -0.017      -0.006
    statefip_19[T.True]    -0.0149      0.003     -4.669      0.000      -0.021      -0.009
    statefip_20[T.True]    -0.0048      0.003     -1.397      0.162      -0.012       0.002
    statefip_21[T.True]     0.0134      0.003      3.899      0.000       0.007       0.020
    statefip_22[T.True]    -0.0021      0.003     -0.640      0.522      -0.008       0.004
    statefip_23[T.True]     0.0102      0.005      2.183      0.029       0.001       0.019
    statefip_24[T.True]    -0.0108      0.003     -3.621      0.000      -0.017      -0.005
    statefip_25[T.True]    -0.0016      0.003     -0.556      0.578      -0.007       0.004
    statefip_26[T.True]     0.0123      0.003      4.457      0.000       0.007       0.018
    statefip_27[T.True]    -0.0082      0.003     -2.828      0.005      -0.014      -0.003
    statefip_28[T.True]     0.0105      0.004      2.641      0.008       0.003       0.018
    statefip_29[T.True]    -0.0050      0.003     -1.683      0.092      -0.011       0.001
    statefip_30[T.True]    -0.0135      0.005     -2.957      0.003      -0.022      -0.005
    statefip_31[T.True]    -0.0117      0.004     -3.170      0.002      -0.019      -0.004
    statefip_32[T.True]    -0.0209      0.004     -4.775      0.000      -0.029      -0.012
    statefip_33[T.True]    -0.0060      0.004     -1.363      0.173      -0.015       0.003
    statefip_34[T.True]    -0.0190      0.003     -6.983      0.000      -0.024      -0.014
    statefip_35[T.True]    -0.0155      0.004     -3.736      0.000      -0.024      -0.007
    statefip_36[T.True] -2.632e-05      0.003     -0.010      0.992      -0.005       0.005
    statefip_37[T.True]    -0.0034      0.003     -1.142      0.253      -0.009       0.002
    statefip_38[T.True]    -0.0185      0.005     -3.645      0.000      -0.028      -0.009
    statefip_39[T.True]     0.0006      0.003      0.240      0.810      -0.005       0.006
    statefip_40[T.True]     0.0165      0.004      4.690      0.000       0.010       0.023
    statefip_41[T.True]     0.0170      0.003      5.068      0.000       0.010       0.024
    statefip_42[T.True]    -0.0019      0.003     -0.701      0.483      -0.007       0.003
    statefip_44[T.True]    -0.0031      0.005     -0.644      0.520      -0.013       0.006
    statefip_45[T.True]    -0.0017      0.003     -0.477      0.633      -0.008       0.005
    statefip_46[T.True]    -0.0245      0.005     -4.920      0.000      -0.034      -0.015
    statefip_47[T.True]     0.0040      0.003      1.256      0.209      -0.002       0.010
    statefip_48[T.True]    -0.0220      0.003     -8.628      0.000      -0.027      -0.017
    statefip_49[T.True]     0.0013      0.004      0.321      0.748      -0.006       0.009
    statefip_50[T.True]     0.0015      0.006      0.255      0.799      -0.010       0.013
    statefip_51[T.True]    -0.0111      0.003     -3.867      0.000      -0.017      -0.005
    statefip_53[T.True]     0.0118      0.003      3.900      0.000       0.006       0.018
    statefip_54[T.True]     0.0213      0.004      5.039      0.000       0.013       0.030
    statefip_55[T.True]    -0.0148      0.003     -5.116      0.000      -0.020      -0.009
    statefip_56[T.True]    -0.0142      0.005     -2.620      0.009      -0.025      -0.004
    educ                   -0.0227      0.000   -161.869      0.000      -0.023      -0.022
    age                     0.0043      4e-05    106.984      0.000       0.004       0.004
    age_sq               1.016e-05   3.58e-06      2.841      0.004    3.15e-06    1.72e-05
    ==============================================================================
    Omnibus:                  1549146.242   Durbin-Watson:                   1.920
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         24833964.724
    Skew:                           3.981   Prob(JB):                         0.00
    Kurtosis:                      18.625   Cond. No.                     4.31e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors are heteroscedasticity robust (HC1)
    [2] The condition number is large, 4.31e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    2SLS Second Stage Results (Manual):
    

    C:\Users\SHAYAN\anaconda3\Lib\site-packages\numpy\core\fromnumeric.py:88: RuntimeWarning: overflow encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    

                                WLS Regression Results                            
    ==============================================================================
    Dep. Variable:               disabwrk   R-squared:                       0.017
    Model:                            WLS   Adj. R-squared:                  0.017
    Method:                 Least Squares   F-statistic:                     411.3
    Date:                Thu, 05 Jun 2025   Prob (F-statistic):               0.00
    Time:                        10:49:13   Log-Likelihood:                    inf
    No. Observations:             1938040   AIC:                              -inf
    Df Residuals:                 1937980   BIC:                              -inf
    Df Model:                          59                                         
    Covariance Type:                  HC1                                         
    =======================================================================================
                              coef    std err          z      P>|z|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept               1.4480      0.100     14.483      0.000       1.252       1.644
    race_2[T.True]          0.0316      0.008      3.943      0.000       0.016       0.047
    race_3[T.True]          0.0094      0.015      0.632      0.528      -0.020       0.038
    race_4[T.True]         -0.0098      0.007     -1.383      0.167      -0.024       0.004
    race_5[T.True]          0.0069      0.011      0.633      0.527      -0.015       0.028
    race_6[T.True]         -0.0106      0.005     -2.160      0.031      -0.020      -0.001
    race_7[T.True]         -0.0771      0.020     -3.928      0.000      -0.116      -0.039
    statefip_2[T.True]      0.0120      0.012      1.020      0.308      -0.011       0.035
    statefip_4[T.True]      0.0139      0.006      2.403      0.016       0.003       0.025
    statefip_5[T.True]      0.0239      0.004      5.372      0.000       0.015       0.033
    statefip_6[T.True]      0.0317      0.007      4.250      0.000       0.017       0.046
    statefip_8[T.True]      0.0393      0.012      3.347      0.001       0.016       0.062
    statefip_9[T.True]      0.0201      0.010      2.084      0.037       0.001       0.039
    statefip_10[T.True]     0.0148      0.008      1.762      0.078      -0.002       0.031
    statefip_11[T.True]     0.0692      0.018      3.799      0.000       0.034       0.105
    statefip_12[T.True]     0.0069      0.005      1.440      0.150      -0.002       0.016
    statefip_13[T.True]     0.0119      0.003      3.613      0.000       0.005       0.018
    statefip_15[T.True]     0.0306      0.007      4.288      0.000       0.017       0.045
    statefip_16[T.True]     0.0207      0.008      2.657      0.008       0.005       0.036
    statefip_17[T.True]     0.0111      0.006      1.776      0.076      -0.001       0.023
    statefip_18[T.True]    -0.0063      0.003     -1.951      0.051      -0.013    2.89e-05
    statefip_19[T.True]     0.0063      0.007      0.956      0.339      -0.007       0.019
    statefip_20[T.True]     0.0254      0.009      2.831      0.005       0.008       0.043
    statefip_21[T.True]    -0.0047      0.006     -0.770      0.441      -0.017       0.007
    statefip_22[T.True]     0.0025      0.004      0.718      0.473      -0.004       0.009
    statefip_23[T.True]     0.0213      0.006      3.784      0.000       0.010       0.032
    statefip_24[T.True]     0.0208      0.009      2.269      0.023       0.003       0.039
    statefip_25[T.True]     0.0333      0.010      3.336      0.001       0.014       0.053
    statefip_26[T.True]     0.0314      0.006      5.294      0.000       0.020       0.043
    statefip_27[T.True]     0.0274      0.010      2.686      0.007       0.007       0.047
    statefip_28[T.True]     0.0093      0.004      2.314      0.021       0.001       0.017
    statefip_29[T.True]     0.0070      0.004      1.563      0.118      -0.002       0.016
    statefip_30[T.True]     0.0194      0.010      1.920      0.055      -0.000       0.039
    statefip_31[T.True]     0.0194      0.009      2.082      0.037       0.001       0.038
    statefip_32[T.True]    -0.0026      0.007     -0.388      0.698      -0.016       0.010
    statefip_33[T.True]     0.0197      0.008      2.364      0.018       0.003       0.036
    statefip_34[T.True]     0.0053      0.007      0.734      0.463      -0.009       0.019
    statefip_35[T.True]    -0.0014      0.006     -0.247      0.805      -0.013       0.010
    statefip_36[T.True]     0.0240      0.007      3.399      0.001       0.010       0.038
    statefip_37[T.True]     0.0019      0.003      0.587      0.557      -0.005       0.008
    statefip_38[T.True]     0.0087      0.009      0.966      0.334      -0.009       0.026
    statefip_39[T.True]     0.0113      0.004      2.837      0.005       0.003       0.019
    statefip_40[T.True]     0.0310      0.005      5.817      0.000       0.021       0.041
    statefip_41[T.True]     0.0491      0.009      5.213      0.000       0.031       0.068
    statefip_42[T.True]     0.0123      0.005      2.608      0.009       0.003       0.022
    statefip_44[T.True]     0.0064      0.006      1.151      0.250      -0.004       0.017
    statefip_45[T.True]    -0.0030      0.004     -0.842      0.400      -0.010       0.004
    statefip_46[T.True]     0.0016      0.009      0.188      0.851      -0.015       0.019
    statefip_47[T.True]    -0.0004      0.003     -0.111      0.912      -0.007       0.006
    statefip_48[T.True]    -0.0157      0.003     -5.041      0.000      -0.022      -0.010
    statefip_49[T.True]     0.0329      0.010      3.451      0.001       0.014       0.052
    statefip_50[T.True]     0.0281      0.009      2.982      0.003       0.010       0.047
    statefip_51[T.True]     0.0082      0.006      1.361      0.174      -0.004       0.020
    statefip_53[T.True]     0.0459      0.010      4.671      0.000       0.027       0.065
    statefip_54[T.True]     0.0097      0.005      1.814      0.070      -0.001       0.020
    statefip_55[T.True]     0.0087      0.007      1.231      0.218      -0.005       0.022
    statefip_56[T.True]     0.0134      0.009      1.432      0.152      -0.005       0.032
    educ_hat               -0.0675      0.012     -5.493      0.000      -0.092      -0.043
    age                     0.0018      0.001      2.761      0.006       0.001       0.003
    age_sq               4.408e-06   3.94e-06      1.119      0.263   -3.32e-06    1.21e-05
    ==============================================================================
    Omnibus:                  1588784.956   Durbin-Watson:                   1.903
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         27018533.243
    Skew:                           4.109   Prob(JB):                         0.00
    Kurtosis:                      19.341   Cond. No.                     3.19e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors are heteroscedasticity robust (HC1)
    [2] The condition number is large, 3.19e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.
    2SLS Results (IV2SLS):
                              IV-2SLS Estimation Summary                          
    ==============================================================================
    Dep. Variable:               disabwrk   R-squared:                     -0.0383
    Estimator:                    IV-2SLS   Adj. R-squared:                -0.0384
    No. Observations:             1938040   F-statistic:                 2.351e+04
    Date:                Thu, Jun 05 2025   P-value (F-stat)                0.0000
    Time:                        10:50:29   Distribution:                 chi2(59)
    Cov. Estimator:                robust                                         
                                                                                  
                                  Parameter Estimates                              
    ===============================================================================
                 Parameter  Std. Err.     T-stat    P-value    Lower CI    Upper CI
    -------------------------------------------------------------------------------
    Intercept       1.4480     0.1028     14.082     0.0000      1.2464      1.6495
    age             0.0018     0.0007     2.6840     0.0073      0.0005      0.0032
    age_sq       4.408e-06  4.054e-06     1.0872     0.2769  -3.538e-06   1.235e-05
    race_2          0.0316     0.0082     3.8359     0.0001      0.0155      0.0477
    race_3          0.0094     0.0152     0.6153     0.5384     -0.0205      0.0392
    race_4         -0.0098     0.0076    -1.2902     0.1970     -0.0248      0.0051
    race_5          0.0069     0.0113     0.6112     0.5410     -0.0153      0.0291
    race_6         -0.0106     0.0052    -2.0498     0.0404     -0.0208     -0.0005
    race_7         -0.0771     0.0203    -3.8083     0.0001     -0.1168     -0.0374
    statefip_2      0.0120     0.0122     0.9836     0.3253     -0.0119      0.0358
    statefip_4      0.0139     0.0060     2.3328     0.0197      0.0022      0.0256
    statefip_5      0.0239     0.0045     5.3680     0.0000      0.0152      0.0326
    statefip_6      0.0317     0.0077     4.1386     0.0000      0.0167      0.0467
    statefip_8      0.0393     0.0121     3.2537     0.0011      0.0156      0.0629
    statefip_9      0.0201     0.0099     2.0273     0.0426      0.0007      0.0395
    statefip_10     0.0148     0.0086     1.7242     0.0847     -0.0020      0.0316
    statefip_11     0.0692     0.0187     3.6965     0.0002      0.0325      0.1059
    statefip_12     0.0069     0.0049     1.4072     0.1594     -0.0027      0.0165
    statefip_13     0.0119     0.0033     3.5792     0.0003      0.0054      0.0185
    statefip_15     0.0306     0.0074     4.1292     0.0000      0.0161      0.0451
    statefip_16     0.0207     0.0080     2.5815     0.0098      0.0050      0.0365
    statefip_17     0.0111     0.0064     1.7316     0.0833     -0.0015      0.0237
    statefip_18    -0.0063     0.0033    -1.9208     0.0548     -0.0128      0.0001
    statefip_19     0.0063     0.0068     0.9313     0.3517     -0.0070      0.0197
    statefip_20     0.0254     0.0092     2.7572     0.0058      0.0073      0.0434
    statefip_21    -0.0047     0.0062    -0.7539     0.4509     -0.0168      0.0075
    statefip_22     0.0025     0.0036     0.7096     0.4780     -0.0044      0.0095
    statefip_23     0.0213     0.0057     3.7294     0.0002      0.0101      0.0325
    statefip_24     0.0208     0.0094     2.2086     0.0272      0.0023      0.0394
    statefip_25     0.0333     0.0103     3.2457     0.0012      0.0132      0.0534
    statefip_26     0.0314     0.0061     5.1690     0.0000      0.0195      0.0432
    statefip_27     0.0274     0.0105     2.6143     0.0089      0.0069      0.0480
    statefip_28     0.0093     0.0041     2.3041     0.0212      0.0014      0.0173
    statefip_29     0.0070     0.0046     1.5310     0.1258     -0.0020      0.0160
    statefip_30     0.0194     0.0104     1.8628     0.0625     -0.0010      0.0398
    statefip_31     0.0194     0.0096     2.0265     0.0427      0.0006      0.0381
    statefip_32    -0.0026     0.0069    -0.3743     0.7082     -0.0161      0.0109
    statefip_33     0.0197     0.0086     2.2988     0.0215      0.0029      0.0365
    statefip_34     0.0053     0.0074     0.7149     0.4747     -0.0092      0.0197
    statefip_35    -0.0014     0.0059    -0.2386     0.8114     -0.0129      0.0101
    statefip_36     0.0240     0.0073     3.3118     0.0009      0.0098      0.0383
    statefip_37     0.0019     0.0034     0.5799     0.5620     -0.0046      0.0085
    statefip_38     0.0087     0.0093     0.9385     0.3480     -0.0095      0.0269
    statefip_39     0.0113     0.0041     2.7833     0.0054      0.0033      0.0193
    statefip_40     0.0310     0.0054     5.7071     0.0000      0.0204      0.0416
    statefip_41     0.0491     0.0097     5.0757     0.0000      0.0301      0.0680
    statefip_42     0.0123     0.0048     2.5522     0.0107      0.0029      0.0217
    statefip_44     0.0064     0.0058     1.1073     0.2681     -0.0049      0.0177
    statefip_45    -0.0030     0.0036    -0.8358     0.4033     -0.0100      0.0040
    statefip_46     0.0016     0.0090     0.1818     0.8557     -0.0160      0.0193
    statefip_47    -0.0004     0.0034    -0.1094     0.9128     -0.0071      0.0064
    statefip_48    -0.0157     0.0032    -4.9428     0.0000     -0.0219     -0.0095
    statefip_49     0.0329     0.0098     3.3544     0.0008      0.0137      0.0521
    statefip_50     0.0281     0.0097     2.9020     0.0037      0.0091      0.0471
    statefip_51     0.0082     0.0062     1.3275     0.1843     -0.0039      0.0203
    statefip_53     0.0459     0.0101     4.5451     0.0000      0.0261      0.0657
    statefip_54     0.0097     0.0054     1.7976     0.0722     -0.0009      0.0203
    statefip_55     0.0087     0.0072     1.2000     0.2301     -0.0055      0.0228
    statefip_56     0.0134     0.0096     1.3843     0.1663     -0.0056      0.0323
    educ           -0.0675     0.0126    -5.3410     0.0000     -0.0923     -0.0428
    ===============================================================================
    
    Endogenous: educ
    Instruments: qob1, qob2, qob3
    Robust Covariance (Heteroskedastic)
    Debiased: False
    Hausman Test for Endogeneity:
    

    C:\Users\SHAYAN\anaconda3\Lib\site-packages\numpy\core\fromnumeric.py:88: RuntimeWarning: overflow encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    

    =======================================================================================
                              coef    std err          z      P>|z|      [0.025      0.975]
    ---------------------------------------------------------------------------------------
    Intercept               1.4480      0.099     14.624      0.000       1.254       1.642
    race_2[T.True]          0.0316      0.008      3.981      0.000       0.016       0.047
    race_3[T.True]          0.0094      0.015      0.638      0.524      -0.019       0.038
    race_4[T.True]         -0.0098      0.007     -1.389      0.165      -0.024       0.004
    race_5[T.True]          0.0069      0.011      0.638      0.523      -0.014       0.028
    race_6[T.True]         -0.0106      0.005     -2.176      0.030      -0.020      -0.001
    race_7[T.True]         -0.0771      0.019     -3.964      0.000      -0.115      -0.039
    statefip_2[T.True]      0.0120      0.012      1.028      0.304      -0.011       0.035
    statefip_4[T.True]      0.0139      0.006      2.424      0.015       0.003       0.025
    statefip_5[T.True]      0.0239      0.004      5.446      0.000       0.015       0.032
    statefip_6[T.True]      0.0317      0.007      4.292      0.000       0.017       0.046
    statefip_8[T.True]      0.0393      0.012      3.379      0.001       0.016       0.062
    statefip_9[T.True]      0.0201      0.010      2.105      0.035       0.001       0.039
    statefip_10[T.True]     0.0148      0.008      1.782      0.075      -0.001       0.031
    statefip_11[T.True]     0.0692      0.018      3.837      0.000       0.034       0.105
    statefip_12[T.True]     0.0069      0.005      1.455      0.146      -0.002       0.016
    statefip_13[T.True]     0.0119      0.003      3.660      0.000       0.006       0.018
    statefip_15[T.True]     0.0306      0.007      4.324      0.000       0.017       0.044
    statefip_16[T.True]     0.0207      0.008      2.681      0.007       0.006       0.036
    statefip_17[T.True]     0.0111      0.006      1.794      0.073      -0.001       0.023
    statefip_18[T.True]    -0.0063      0.003     -1.974      0.048      -0.013   -4.36e-05
    statefip_19[T.True]     0.0063      0.007      0.965      0.334      -0.007       0.019
    statefip_20[T.True]     0.0254      0.009      2.860      0.004       0.008       0.043
    statefip_21[T.True]    -0.0047      0.006     -0.778      0.436      -0.016       0.007
    statefip_22[T.True]     0.0025      0.003      0.727      0.467      -0.004       0.009
    statefip_23[T.True]     0.0213      0.006      3.830      0.000       0.010       0.032
    statefip_24[T.True]     0.0208      0.009      2.291      0.022       0.003       0.039
    statefip_25[T.True]     0.0333      0.010      3.369      0.001       0.014       0.053
    statefip_26[T.True]     0.0314      0.006      5.348      0.000       0.020       0.043
    statefip_27[T.True]     0.0274      0.010      2.713      0.007       0.008       0.047
    statefip_28[T.True]     0.0093      0.004      2.345      0.019       0.002       0.017
    statefip_29[T.True]     0.0070      0.004      1.580      0.114      -0.002       0.016
    statefip_30[T.True]     0.0194      0.010      1.938      0.053      -0.000       0.039
    statefip_31[T.True]     0.0194      0.009      2.103      0.036       0.001       0.037
    statefip_32[T.True]    -0.0026      0.007     -0.391      0.696      -0.016       0.010
    statefip_33[T.True]     0.0197      0.008      2.388      0.017       0.004       0.036
    statefip_34[T.True]     0.0053      0.007      0.742      0.458      -0.009       0.019
    statefip_35[T.True]    -0.0014      0.006     -0.249      0.803      -0.012       0.010
    statefip_36[T.True]     0.0240      0.007      3.433      0.001       0.010       0.038
    statefip_37[T.True]     0.0019      0.003      0.594      0.552      -0.004       0.008
    statefip_38[T.True]     0.0087      0.009      0.976      0.329      -0.009       0.026
    statefip_39[T.True]     0.0113      0.004      2.868      0.004       0.004       0.019
    statefip_40[T.True]     0.0310      0.005      5.880      0.000       0.021       0.041
    statefip_41[T.True]     0.0491      0.009      5.264      0.000       0.031       0.067
    statefip_42[T.True]     0.0123      0.005      2.636      0.008       0.003       0.021
    statefip_44[T.True]     0.0064      0.006      1.159      0.246      -0.004       0.017
    statefip_45[T.True]    -0.0030      0.004     -0.853      0.394      -0.010       0.004
    statefip_46[T.True]     0.0016      0.009      0.189      0.850      -0.015       0.019
    statefip_47[T.True]    -0.0004      0.003     -0.112      0.911      -0.007       0.006
    statefip_48[T.True]    -0.0157      0.003     -5.097      0.000      -0.022      -0.010
    statefip_49[T.True]     0.0329      0.009      3.483      0.000       0.014       0.051
    statefip_50[T.True]     0.0281      0.009      3.013      0.003       0.010       0.046
    statefip_51[T.True]     0.0082      0.006      1.375      0.169      -0.003       0.020
    statefip_53[T.True]     0.0459      0.010      4.717      0.000       0.027       0.065
    statefip_54[T.True]     0.0097      0.005      1.837      0.066      -0.001       0.020
    statefip_55[T.True]     0.0087      0.007      1.244      0.214      -0.005       0.022
    statefip_56[T.True]     0.0134      0.009      1.444      0.149      -0.005       0.031
    educ                   -0.0675      0.012     -5.547      0.000      -0.091      -0.044
    resid                   0.0448      0.012      3.681      0.000       0.021       0.069
    age                     0.0018      0.001      2.787      0.005       0.001       0.003
    age_sq               4.408e-06    3.9e-06      1.130      0.259   -3.24e-06    1.21e-05
    =======================================================================================
    First Stage F-Statistic: 1789.4382609583222
    


```python


# Filter valid data
data = data[data['birthqtr'].isin([1, 2, 3, 4]) & data['educ'].notna() & data['incwage'].notna()]

# Compute birth year
data['birthyr'] = 1980 - data['age']

# Step 2: Replicate Figures I, II, III from AK91 (Question 3a)
try:
    # Figure I: Average education by quarter of birth and year of birth
    fig1_data = data.groupby(['birthyr', 'birthqtr'])['educ'].mean().reset_index()
    print("Figure I data sample:", fig1_data.head())
    fig1_pivot = fig1_data.pivot(index='birthyr', columns='birthqtr', values='educ')
    print("Figure I pivot sample:", fig1_pivot.head())

    plt.figure(figsize=(10, 6))
    for q in range(1, 5):
        if q in fig1_pivot.columns:
            plt.plot(fig1_pivot.index, fig1_pivot[q], label=f'Q{q}')
    plt.xlabel('Year of Birth')
    plt.ylabel('Average Years of Education')
    plt.title('Average Education by Quarter of Birth and Year of Birth (Figure I)')
    plt.legend()
    plt.grid(True)
    plt.savefig('figure1.png')
    plt.show()  # Display plot
    # plt.close()  # Commented out to keep plot open

    # Figure II: Difference in education (Q1 vs. Q4)
    fig2_data = fig1_pivot[1] - fig1_pivot[4]
    plt.figure(figsize=(10, 6))
    plt.plot(fig2_data.index, fig2_data, label='Q1 - Q4')
    plt.xlabel('Year of Birth')
    plt.ylabel('Difference in Average Education (Q1 - Q4)')
    plt.title('Education Difference by Quarter of Birth (Figure II)')
    plt.legend()
    plt.grid(True)
    plt.savefig('figure2.png')
    plt.show()
    # plt.close()

    # Figure III: Education by age and QOB
    fig3_data = data.groupby(['age', 'birthqtr'])['educ'].mean().reset_index()
    fig3_pivot = fig3_data.pivot(index='age', columns='birthqtr', values='educ')
    plt.figure(figsize=(10, 6))
    for q in range(1, 5):
        if q in fig3_pivot.columns:
            plt.plot(fig3_pivot.index, fig3_pivot[q], label=f'Q{q}')
    plt.xlabel('Age')
    plt.ylabel('Average Years of Education')
    plt.title('Average Education by Age and Quarter of Birth (Figure III)')
    plt.legend()
    plt.grid(True)
    plt.savefig('figure3.png')
    plt.show()
    # plt.close()
except Exception as e:
    print(f"Error in plotting education figures: {e}")

# Step 3: Analyze First-Quarter Effect (Question 3b)
qob_educ = data.groupby('birthqtr')['educ'].mean()
print("Average Education by Quarter of Birth:")
print(qob_educ)

# Step 4: Repeat Figures for Wages (Question 3c)
try:
    # Figure I: Average wages by quarter of birth and year of birth
    fig1_wage_data = data.groupby(['birthyr', 'birthqtr'])['incwage'].mean().reset_index()
    fig1_wage_pivot = fig1_wage_data.pivot(index='birthyr', columns='birthqtr', values='incwage')

    plt.figure(figsize=(10, 6))
    for q in range(1, 5):
        if q in fig1_wage_pivot.columns:
            plt.plot(fig1_wage_pivot.index, fig1_wage_pivot[q], label=f'Q{q}')
    plt.xlabel('Year of Birth')
    plt.ylabel('Average Wage Income')
    plt.title('Average Wages by Quarter of Birth and Year of Birth')
    plt.legend()
    plt.grid(True)
    plt.savefig('figure1_wages.png')
    plt.show()
    # plt.close()

    # Figure II: Difference in wages (Q1 vs. Q4)
    fig2_wage_data = fig1_wage_pivot[1] - fig1_wage_pivot[4]
    plt.figure(figsize=(10, 6))
    plt.plot(fig2_wage_data.index, fig2_wage_data, label='Q1 - Q4')
    plt.xlabel('Year of Birth')
    plt.ylabel('Difference in Average Wages (Q1 - Q4)')
    plt.title('Wage Difference by Quarter of Birth')
    plt.legend()
    plt.grid(True)
    plt.savefig('figure2_wages.png')
    plt.show()
    # plt.close()

    # Figure III: Wages by age and QOB
    fig3_wage_data = data.groupby(['age', 'birthqtr'])['incwage'].mean().reset_index()
    fig3_wage_pivot = fig3_wage_data.pivot(index='age', columns='birthqtr', values='incwage')
    plt.figure(figsize=(10, 6))
    for q in range(1, 5):
        if q in fig3_wage_pivot.columns:
            plt.plot(fig3_wage_pivot.index, fig3_wage_pivot[q], label=f'Q{q}')
    plt.xlabel('Age')
    plt.ylabel('Average Wage Income')
    plt.title('Average Wages by Age and Quarter of Birth')
    plt.legend()
    plt.grid(True)
    plt.savefig('figure3_wages.png')
    plt.show()
    # plt.close()
except Exception as e:
    print(f"Error in plotting wage figures: {e}")
```

    Figure I data sample:    birthyr  birthqtr      educ
    0     1930         1  5.654815
    1     1930         2  5.689936
    2     1930         3  5.695324
    3     1930         4  5.736146
    4     1931         1  5.773065
    Figure I pivot sample: birthqtr         1         2         3         4
    birthyr                                         
    1930      5.654815  5.689936  5.695324  5.736146
    1931      5.773065  5.724192  5.793253  5.827098
    1932      5.778240  5.838626  5.858044  5.890825
    1933      5.830609  5.750356  5.861634  5.852297
    1934      5.885209  5.850739  5.894898  5.910725
    


    
![png](output_4_1.png)
    



    
![png](output_4_2.png)
    



    
![png](output_4_3.png)
    


    Average Education by Quarter of Birth:
    birthqtr
    1    6.424040
    2    6.479475
    3    6.506075
    4    6.500461
    Name: educ, dtype: float64
    


    
![png](output_4_5.png)
    



    
![png](output_4_6.png)
    



    
![png](output_4_7.png)
    



```python

```
