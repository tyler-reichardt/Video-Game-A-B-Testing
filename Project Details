
## Usage/Examples

```javascript
import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

df = pd.read_csv('/kaggle/input/mobile-games-ab-testing/cookie_cats.csv')
df.head()



df.dtypes



df.isna().sum()



print(df['retention_1'].value_counts())
print(df['retention_7'].value_counts())



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



# Create a boxplot
fig, ax = plt.subplots()
ax.boxplot(df['sum_gamerounds'])

# Add labels and title
ax.set_xlabel('Data')
ax.set_ylabel('Value')
ax.set_title('Boxplot of Data')

# Display the chart
plt.show()



median = df['sum_gamerounds'].median()
q1, q3 = df['sum_gamerounds'].describe()[['25%', '75%']]
iqr = q3 - q1
outliers = df['sum_gamerounds'][(df['sum_gamerounds'] < q1 - 1.5 * iqr) | (df['sum_gamerounds'] > q3 + 1.5 * iqr)].values

print('medium value: ', median)
print('Quartile 1 value: ' + str(q1))
print('Quartile 3 value: ' + str(q3))
print('Inter Quartile Range (IQR) value: ' + str(iqr))
print('Outliers (Ascending): ' + str(np.sort(outliers)))



# A/B Groups & Target Summary Stats
df.groupby("version").sum_gamerounds.agg(["count", "median", "mean", "std", "max"])



df[(df.version == "gate_30")].reset_index().set_index("index").sum_gamerounds.plot(legend = True, label = "Gate 30", figsize = (20,5))
df[df.version == "gate_40"].reset_index().set_index("index").sum_gamerounds.plot(legend = True, label = "Gate 40", alpha = 0.8)
plt.suptitle("After Removing The Extreme Value", fontsize = 20);



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

