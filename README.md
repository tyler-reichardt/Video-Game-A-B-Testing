# Video Game A/B Testing



## Introduction
Statistical Analysis of two different gaming interfaces and their impact on retention.

This project aims to test whether a new video game interface results in higher retention in both 1 and 7 days. We used A/B Testing methodology with Python to analyze the results of the experiment.

## Technologies
#### 1. Jupyter Notebook
#### 2. Python 


## Key Features
Here, describe the key features of your project. This section should highlight what makes your project unique and valuable to potential users.

## Installation
Dependencies for this project are: 

 - pandas
 - numpy
 - matplotlib.pyplot
 - scipy.stats




## Deployment

## To install the libraries in the notebook.

```bash
import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
```

## Creating a Pandas dataframe from the CSV file.

```bash
df = pd.read_csv('/kaggle/input/mobile-games-ab-testing/cookie_cats.csv')
df.head()
```
![DataFrame](AB%20Testing/dataframe.png)


## Understanding the data types within the dataframe.

```bash
df.dtypes
```

![DataTypes](AB%20Testing/datatypes.png)


## Checking for Null values.

```bash
df.isna().sum()
```

![Null Count](AB%20Testing/dataframe_null_count.png)


## Getting an Idea of the Retention for 1 and 7 days.

```bash
print(df['retention_1'].value_counts())
print(df['retention_7'].value_counts())
```

![Retention](AB%20Testing/retention_value_counts.png)


## Visualizing Retention for 1 and 7 days.

```bash
# Define the data
retention_1 = df['retention_1'].value_counts()
retention_7 = df['retention_7'].value_counts()
labels = ['False', 'True']

# Create the figure and axis objects
fig, ax = plt.subplots()

# Create the stacked bar chart
ax.bar(labels, retention_1, label='Retention 1 day')
ax.bar(labels, retention_7, bottom=retention_1, label='Retention 7 days')

# Add labels and legend
ax.set_xlabel('Retention status')
ax.set_ylabel('Number of users')
ax.set_title('Retention by duration')
ax.legend()

# Display the chart
plt.show()
```

![Chart1](AB%20Testing/retention_value_counts_by_graph.png)


## Understanding the the distribution of the data and identifying outliers.

```bash
# Create a boxplot
fig, ax = plt.subplots()
ax.boxplot(df['sum_gamerounds'])

# Add labels and title
ax.set_xlabel('Data')
ax.set_ylabel('Value')
ax.set_title('Boxplot of Data')

# Display the chart
plt.show()
```

![Chart2](AB%20Testing/sum_gamerounds_boxplot.png)


## Defining statistical variables to remove outliers.


```bash
median = df['sum_gamerounds'].median()
q1, q3 = df['sum_gamerounds'].describe()[['25%', '75%']]
iqr = q3 - q1
outliers = df['sum_gamerounds'][(df['sum_gamerounds'] < q1 - 1.5 * iqr) | (df['sum_gamerounds'] > q3 + 1.5 * iqr)].values

print('medium value: ', median)
print('Quartile 1 value: ' + str(q1))
print('Quartile 3 value: ' + str(q3))
print('Inter Quartile Range (IQR) value: ' + str(iqr))
print('Outliers (Ascending): ' + str(np.sort(outliers)))
```

![Chart3](AB%20Testing/sum_gamerounds_stats.png)

### Question: 
Why use a multiplier of 1.5?
### Answer:
 We use the 1.5 x quartiles in statistics to remove outliers because it is a common and simple method to identify extreme values that are significantly different from the rest of the data. The 1.5 x interquartile range (IQR) method is based on the concept that outliers lie outside the range of typical values, which can be defined as the middle 50% of the data. By multiplying the IQR by 1.5, we can establish a threshold beyond which values are considered outliers and removed from the dataset. This method is relatively robust and works well for many types of data distributions, but it is not a perfect method and should be used in conjunction with other outlier detection techniques when appropriate.


## After removing outliers, lets take a look at the profile of the data distribution.

```bash
# A/B Groups & Target Summary Stats
df.groupby("version").sum_gamerounds.agg(["count", "median", "mean", "std", "max"])
```

![Chart4](AB%20Testing/groupby_stats_screenshot_not_describe.png)


## Furthermore, lets see how to data is spread.

```bash
# Filter out the max value within the dataset.
df = df[df.sum_gamerounds < df.sum_gamerounds.max()]

# Summary Stats: sum_gamerounds
df.describe([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])[["sum_gamerounds"]].T
```

![Chart5](AB%20Testing/groupby_stats_screenshot.png)

## This code is plotting two line graphs of the sum of game rounds played for different versions of a game after removing extreme values and adding a super title.

```bash
# Plot game rounds for gate_30
df[(df.version == "gate_30")].reset_index().set_index("index").sum_gamerounds.plot(legend = True, label = "Gate 30", figsize = (20,5))

# Plot game rounds for gat_40
df[df.version == "gate_40"].reset_index().set_index("index").sum_gamerounds.plot(legend = True, label = "Gate 40", alpha = 0.8)

# Plot Title
plt.suptitle("After Removing The Extreme Value", fontsize = 20);
```
![Chart6](AB%20Testing/sum_gamerounds_plot.png)


## This Python function performs an A/B test on a given dataset to compare the sum of game rounds played between two different versions of a game and determine if there is a statistically significant difference, using normality and variance tests and a t-test or Mann-Whitney U test.

```bash
def AB_Test(df, pthres=0.05):
    # H0:  There is no statistical difference between the gate_30 and the gate_40.
    print(df.groupby('version').agg({"sum_gamerounds": ["count", "mean"]}))

    print(5 * "*" + " NORMAL DISTRIBUTION ASSUMPTION " + 5 * "*" + "\n")
    # H0 : The compared groups have a normal distribution
    pvalue_gate_30 = shapiro([df["version"] == "gate_30"])[1]
    pvalue_gate_40 = shapiro([df["version"] == "gate_40"])[1]
    print('p-value_gate30 = %.5f' % (pvalue_gate_30))
    print('p-value_gate40 = %.5f' % (pvalue_gate_40))

    if (pvalue_gate_30 < pthres) & (pvalue_gate_40 < pthres):
        print("Normality H0 Hypothesis Rejected.\n\n")
    else:
        print("Normality H0 Hypothesis Not Rejected.\n")

    print(5 * "*" + " VARIANCE HOMOGENEOUS ASSUMPTION " + 5 * "*" + "\n")
    # H0 : The compared groups have equal variance.
    p_value_levene = levene(df.loc[df["version"] == "gate_30", "sum_gamerounds"],
                            df.loc[df["version"] == "gate_40", "sum_gamerounds"])[1]
    print('p_value_levene = %.5f' % p_value_levene)

    if p_value_levene < pthres:
        print("Variance Homogeneity H0 Hypothesis Rejected.\n")
    else:
        print("Variance Homogeneity H0 Hypothesis Not Rejected.\n")

    if ((pvalue_gate_30 > pthres) & (pvalue_gate_40 > pthres)) & (p_value_levene > pthres):
        p_value_ttest = ttest_ind(df.loc[df["version"] == "gate_30"],
                                  df.loc[df["version"] == "gate_40"],
                                  equal_var=True)[1]
        print('p_value_ttest = %.5f' % p_value_ttest)

    elif ((pvalue_gate_30 > pthres) & (pvalue_gate_40 > pthres)) & (p_value_levene < pthres):
        p_value_ttest = ttest_ind(df.loc[df["version"] == "gate_30"],
                                  df.loc[df["version"] == "gate_40"],
                                  equal_var=False)[1]
        print('p_value_ttest = %.5f' % p_value_ttest)
    else:
        print("Non-Parametric Test should be done.\n\n")
        pvalue = mannwhitneyu(df.loc[df["version"] == "gate_30", "sum_gamerounds"],
                              df.loc[df["version"] == "gate_40", "sum_gamerounds"])[1]
        print('p_value = %.5f' % pvalue)

    print(5 * "*" + " RESULT " + 5 * "*" + "\n")

    if pvalue < pthres:
        print(
            f"p-value {round(pvalue, 5)} < 0.05  H0 Rejected. That is, there is a statistically significant difference between them.")

    else:
        print(
            f"p-value > {pthres} H0 Not Rejected, That is, there is no statistically significant difference between them. The difference is luck.")
AB_Test(df,0.05)
```
![Chart7](AB%20Testing/sum_gamerounds_A_B_testing.png)













































## Conclusion

### Based on the A/B test result, we cannot reject the null hypothesis that there is no statistically significant difference between the two interfaces of the game, which means that the difference in the number of game rounds played by users is due to chance or luck. Therefore, we cannot conclude that one interface is better than the other.