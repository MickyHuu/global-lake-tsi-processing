This repository contains the Python code used for the analysis and visualization of long-term global lake trophic state index (TSI) records from 1984 to 2023.


### `01_mapping_global_TSI.py`
Plots the spatial distribution of long-term mean lake TSI at the global scale.
Main inputs:
- Lake-level long-term mean TSI table
- Global country boundary shapefile
Main output:
- Global map of long-term mean TSI


### `02_MannKendall_test.py`
Calculates lake-level long-term temporal trends using:
- Theil–Sen slope
- Mann–Kendall trend test
Main outputs include:
- Sen slope
- Sen intercept
- Mann–Kendall S statistic
- Mann–Kendall Z statistic
- Mann–Kendall p value
- Number of valid years
- Trend classification
Trend classes are:
- Significant decrease
- Significant increase
- Not significant


### `03_segmented_regression.py`
Performs global-scale breakpoint analysis on standardized lake TSI time series.
The script:
1. standardizes each lake time series using a z-score;
2. calculates the annual global mean z-score and its 95% confidence interval;
3. fits a one-break segmented regression;
4. searches for the breakpoint by minimizing residual sum of squares;
5. compares the segmented model with a simple linear model using R², AIC, and BIC;
6. estimates breakpoint uncertainty using a moving-block residual bootstrap;
7. reports pre-break and post-break slopes and their uncertainty.
Main outputs include:
- Estimated breakpoint
- Breakpoint 95% confidence interval
- Pre-break and post-break slopes
- Linear and segmented model statistics
- Bootstrap breakpoint distribution
- Breakpoint profile
- Global breakpoint figures


### `04_plot_continent_breakpoints.py`
Visualizes continental-scale TSI breakpoint results for:
- North America
- Europe
- Asia
- Africa
- South America
- Oceania

