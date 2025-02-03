import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# Create a Pandas DataFrame from iris data
iris = datasets.load_iris()
irisdf: pd.DataFrame = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Add the target variable & its target_names value to the DataFrame as Species column
irisdf['species'] = iris.target_names[iris.target]

#Q1.a: Make a histogram of the variable Sepal.Width
sepal_widths = irisdf['sepal width (cm)'].sort_values()
plt.hist(sepal_widths, edgecolor='black', bins=25)
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.title('Sepal Width Histogram')
plt.show()
plt.clf() # Reset pyplot object before next question

#Q1.b & c: Based on the histogram from #1a, which would you expect to be higher, the mean or the median? Why?
# Confirm your answer by actually finding these values.
sw_mean = sepal_widths.mean()
sw_median = sepal_widths.median()
print("Sepal width mean =",f"{sw_mean:.2f}","; sepal width median", sw_median)

#More exploration of Q1 skew
plt.boxplot(sepal_widths, vert=False, showmeans=True, meanline=True)
plt.title('Sepal Width Boxplot')
plt.xlabel('Sepal Width (cm)')
plt.show()

#Q1.d: Only 27% of the flowers have a Sepal.Width higher than ________ cm
# Calculate the sepal width that 27% of the sepals are wider than.  if only 27% are wider, 100-27 = 73% are as or more
# narrow (or, the 73rd percentile)
percentile = 73
width_boundary = np.percentile(sepal_widths, percentile)
print("Only 27% of the flowers have a Sepal.Width greater than:",f"{width_boundary:.2f}"," cm")

'''y = np.quantile(sepal_widths, .73, method="linear")
y2 = np.quantile(sepal_widths, .73, method="inverted_cdf")
y3 = np.quantile(sepal_widths, .73, method="closest_observation")
print("Some alternate calcs of percentile:", f"{y:.2f}", f"{y2:.2f}", f"{y3:.2f}")'''

##double check?
def percentile_from_mean_std(p_mean, p_std, p_percentile):
    """Calculates the value at a given percentile for a normal distribution."""
    return norm.ppf(p_percentile / 100, loc=p_mean, scale=p_std)

value = percentile_from_mean_std(sw_mean, sepal_widths.std(), percentile)
print(f"Using SciPy norm.ppf, the value at the {percentile}th percentile is: {value:.2f}")

#Q1.e:Make scatterplots of each pair of the numerical variables in iris (There should be 6 pairs/plots).
# Create a pairplot using seaborn
sns.pairplot(irisdf, vars=irisdf.columns[:4], kind="scatter", corner=True)
plt.show()

'''---------------------------------'''
#Q2
#Import PlantGrowth data
data = { "weight": [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26], "group": ["ctrl"] * 10 + ["trt1"] * 10 + ["trt2"] * 10}
PlantGrowth = pd.DataFrame(data)

#Q2.a:Make a histogram of the variable weight with breakpoints (bin edges) at every 0.3 units, starting at 3.3.
bin_start = 3.3
bin_step = 0.3
weights = PlantGrowth['weight']
#make sure we include all values in histogram
max_edge = weights.max() + bin_step
xbins = np.arange(bin_start, max_edge, bin_step)
style = {'edgecolor': 'black', 'linewidth': 2}

fig, ax = plt.subplots()
ax.hist(weights, bins=xbins, **style)
ax.set_ylabel('Frequency')
ax.set_xlabel('Weight (0.3 unit bins)')
ax.set_title('Histogram of Plant Weights')
plt.show()
plt.clf()

#Q2.b:Make boxplots of weight separated by group in a single graph.
#this was much easier in Seaborn
sns.boxplot(data=PlantGrowth, x="group", y="weight")
plt.title('Boxplots of Plant Weights by Group')
plt.show()

#Q2.d: Find the exact percentage of the "trt1" weights that are below the minimum "trt2" weight.
trt2_weights = PlantGrowth['weight'][PlantGrowth['group']=='trt2']
trt2_min = np.min(trt2_weights)
trt1_weights = PlantGrowth['weight'][PlantGrowth['group']=='trt1']
trt1_count = trt1_weights.count()
#get count of just those trt1 weights less than min trt2 weight
trt1_belowtrt2_count = trt1_weights[trt1_weights < trt2_min].count()
percentage_trt1_below_trt2 = (trt1_belowtrt2_count/trt1_count)*100
print("Percentage of trt1 weights less than min trt2 weight:", percentage_trt1_below_trt2)

#Q2.e Only including plants with a weight above 5.5, make a barplot of the variable group.
# Make the barplot colorful using some color palette
## Plot 1 - did not like
'''weight_limit = 5.5
hw_plants = PlantGrowth[PlantGrowth['weight'] > weight_limit]
#print("Plants above 5.5:",hw_plants)
hw_groups = hw_plants['group']
hw_weights = hw_plants['weight']
sns.barplot(x=hw_groups, y=hw_weights, palette="viridis")
plt.ylim(weight_limit) #start plot at 5.5
plt.title('Bar Plot of High-Weight Plants by Group')
plt.show()'''

##Plot 2 - told a clearer story
weight_limit = 5.5
hw_plants = PlantGrowth[PlantGrowth['weight'] > weight_limit].sort_values(by='group')
frequency_table = hw_plants['group'].value_counts().sort_values()
labels = frequency_table.index
values = frequency_table.values

# Create the bar plot
sns.barplot(x=labels, y=values, palette="viridis")
plt.title("Bar Plot of High-Weight Plants by Group")
plt.xlabel("Group")
plt.ylabel("Count")
plt.show()
