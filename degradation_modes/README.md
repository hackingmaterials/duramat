# Degradation modes analysis

## Setup

1. Git clone the duramat repository
```
$ git clone https://github.com/hackingmaterials/duramat.git
```
2. Set the following environment variables for Plotly:
	* 'PLOTLY_USERNAME' : Username from [Plotly](https://plot.ly/accounts/login/?action=signup)
    * 'PLOTLY_KEY' : API key from [Plotly](https://plot.ly/accounts/login/?action=signup)
3. If not provided already, obtain Dirk Jordan's Rd_review_8.csv file and replace the following code at the top of each Jupyter notebook:

Code to be replaced:
```python
sys.path.append('../data/NREL')
import retrieve_data as rd
solar = rd.retrieve_dirks_sheet()
```
Replace with:
```
solar = pd.read_csv('/path/to/file.csv', encoding='utf-8')
```

Once these steps have been completed, each of the three Jupyter analysis notebooks can be run.


## Classification models analysis

### Summary
This notebook is used for exploring Decision Trees and Naive Bayes models to find trends in the data. This notebook operates in the following fashion:

1. Import data and clean
2. Prepare the pandas DataFrame that will be used for building both classification models, this involves:
	* Creating one hot encoding variables for the categorical data, climate and mounting
	* Bin installation year data into 'Before 2000 and After 2000' categories
	* Include the previously created one hot encoding variables of degradation modes presence into this model
3. Build decision trees using mounting type and climate as features and each degradation mode as a target
4. Export them to the output/ directory as dot and png files
5. Show Scikit accuracy score for each decision tree model built
5. OPTIONAL: Exemplifies the code used for optimizing the parameters used to build the decision tree models
6. Build Naive Bayes models using the same DataFrame with mounting type, climate and installation year as features and each degradation mode as a target
7. Show the Scikit score for each Naive Bayes model built
8. Perform analysis using these Naive Bayes models

*Pydocs for all contained functions can be found in the docs file*

## Climate and mounting analysis

### Summary
This notebook is used for exploring the effects of mounting type on degradation modes as well as the most detrimental combinations of mounting and climate class for PV modules. This notebook operates in the following fashion:

1. Import data and clean
2. Create a DataFrame that tracks weighted percentage[^1] of each degradation mode for modules with each of the four selected mounting types (roof rack, 1-axis tracker, single-axis, rack)
3. Graph the DataFrame as a grouped bar chart using Plotly
4. Repeat steps 2 & 3 for data divided into each of the four selected climate types (moderate, hot & humid, desert, snow)
5. Repeat steps 2-4, calculating unweighted percentage[^2] instead
6. Create categorical dot plots for weighted and unweighted percentages of degradation seen in all climates

*Pydocs for all contained functions can be found in the docs file*

## Correlating modes analysis

### Summary
This notebook is used for exploring the strength of correlation between pairs of degradation modes observed on PV modules. This notebook operates in the following fashion:

1. Import data and clean
2. Create a DataFrame that holds the weighted[^1] correlation strength of all pairs of degradation modes by using the following formula:

```
P(Degradation mode A & Degradation mode B) / P(Degradation mode A)P(Degradation mode B)
```

3. Create a Plotly heatmap of the resulting DataFrame
4. Repeat steps 2-3 twice and instead use data split into two categories: installation year before 2000 and after 2000
5. Repeat steps 2-4, calculating the unweighted correlation strength instead[^2]

*Pydocs for all contained functions can be found in the docs file*

## Future work
In terms of future work here are some suggested directions/back log items:

* Investigate the low recommended max_depth of the decision tree classifcation models
* Build a logistic regression classification model on the data
* Explore the effects of vairables such as Voc, FF and Pmax on degradation
* Build a ROC curve for the Naive Bayes models to determine the best threshold for predicting degradation using the model

[^1]: In situations where there is a system of modules reported, the *weighted* percentage calculation will count all of the modules within the system as degrading and add to the total count of modules: (# of degrading systems * # of modules in each) / (# of total systems * # of modules in each)
[^2]: In situations where there is a system of modules reported, the *unweighted* percentage calculation will count all degrading systems as a single module and add one the total count of modules: (# of degrading systems) / (# of total systems)





