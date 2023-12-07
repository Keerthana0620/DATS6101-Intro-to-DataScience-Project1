
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

#%%

df_input = pd.read_csv('C:\Keerthu\GW\IntroToDataScience_6101\Project\Data_sets\Food_Deserts_in_US.csv')
df_input.head()

# %%

print(df_input.isnull().sum())
print(df_input.dtypes)

# %%
print(df_input.info())
print(df_input.describe())

# %%

df = df_input.drop('County', axis=1)
df.head()

#%%
dff = df_input.drop(['CensusTract', 'State', 'County'], axis=1)
dff.head()

# %%

def Encoder(x):
    """
    Encoding Categorical variables in the dataset.
    """
    columnsToEncode = list(x.select_dtypes(include=['object']))
    le = preprocessing.LabelEncoder()
    for feature in columnsToEncode:
        try:
           x[feature] = le.fit_transform(x[feature])
        except:
            print('Error encoding '+feature)
    return df

dff = df.copy()
dff = Encoder(dff)
dff.info()

# %%

sns.set(rc={'figure.figsize':(147,147)})
correlation_matrix = dff.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu')
plt.show()
plt.savefig('C:\Keerthu\GW\IntroToDataScience_6101\Project\Data_sets\your_plot.png', dpi=300, transparent=True)

#%%

numlist = dff.columns.to_list()
numlist.remove('CensusTract')
numlist.remove('LILATracts_1And10')

pizzaPreprocessorNum = ColumnTransformer(
    [
        ("numerical", StandardScaler(), numlist),
    ],
    verbose_feature_names_out=False,
    remainder= 'passthrough',
).fit(dff)

df_std = pd.DataFrame(pizzaPreprocessorNum.fit_transform(dff), columns=pizzaPreprocessorNum.fit(dff).get_feature_names_out())
print(df_std.head())

#%%

X = dff.drop('LILATracts_1And10', axis=1)
y = dff['LILATracts_1And10']

#%%

# encoding using getdummies
columns = ['Urban', 'GroupQuartersFlag', 'LILATracts_halfAnd10', 'LILATracts_1And20', 'LILATracts_Vehicle', 'HUNVFlag', 'LowIncomeTracts', 'LA1and10', 'LAhalfand10', 'LA1and20', 'LATracts_half', 'LATracts1', 'LATracts10', 'LATracts20', 'LATractsVehicle_20']
X = pd.get_dummies(X, columns=columns, drop_first=True)

#%%

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
print("X_Train set shape: ", X_train.shape)
print("y_Train set shape: ", y_train.shape)
print("X_test set shape: ", X_test.shape)
print("y_test set shape: ", y_test.shape)

#%%

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%

#################
# Food deserts
################

# Logistic regression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model (you can use different metrics depending on your problem)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Extract feature importance from the trained model
feature_importance = model.coef_[0]
feature_importance = sorted(feature_importance)

# Print feature importance scores
for feature, importance in zip(X.columns, feature_importance):
    print(f"{feature}: {importance}")


# POP2010: -0.47479003051782126
## OHU2010: -0.4225699312330403
# NUMGQTRS: -0.29808685321217154
## PCTGQTRS: -0.161199904517285
## PovertyRate: -0.1438876130872264
## MedianFamilyIncome: -0.14079932098831766
## LAPOP1_10: -0.1226989787345083

# TractNHOPI: 0.10338402370772547
# TractAIAN: 0.10763656026722508
# TractOMultir: 0.11492693203907474
# TractHispanic: 0.12645757859662257
# TractHUNV: 0.1283876718389017
# TractSNAP: 0.13236952100036595
# Urban_1: 0.13447166904753308
# GroupQuartersFlag_1: 0.14516466030523784
# LILATracts_halfAnd10_1: 0.14814552441981466
# LILATracts_1And20_1: 0.15483387946623883
# LILATracts_Vehicle_1: 0.17665556010149638
# HUNVFlag_1: 0.20304750861617096
# LowIncomeTracts_1: 0.3211199102125567
# LA1and10_1: 0.34604805254961885
# LAhalfand10_1: 1.1838385384816028
# LA1and20_1: 1.2345689250373868
# LATracts_half_1: 1.322990651747802
# LATracts1_1: 1.7585562415408522
# LATracts10_1: 2.366560368878645
# LATracts20_1: 2.732991725013855
## LATractsVehicle_20_1: 3.145828709776273

#%%

from statsmodels.formula.api import glm
import statsmodels.api as sm 

model = glm(formula='LILATracts_1And10 ~ OHU2010 + PCTGQTRS + PovertyRate + MedianFamilyIncome + LAPOP1_10 + LILATracts_Vehicle + Urban + LATracts20 + GroupQuartersFlag', data=dff, family=sm.families.Binomial())
model_fit = model.fit()
print( model_fit.summary() )

# Pseudo R2 - higher, deviance - lower


#%%

# importance 

df_importance = dff[['LILATracts_1And10', 'OHU2010', 'PCTGQTRS', 'PovertyRate', 'MedianFamilyIncome', 'LAPOP1_10', 'LILATracts_Vehicle', 'Urban', 'LATracts20', 'GroupQuartersFlag']]
X = df_importance.drop('LILATracts_1And10', axis=1)
y = df_importance['LILATracts_1And10']
columns = ['Urban', 'LILATracts_Vehicle', 'LATracts20']
X = pd.get_dummies(X, columns=columns, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model (you can use different metrics depending on your problem)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Extract feature importance from the trained model
feature_importance = model.coef_[0]
feature_importance = sorted(feature_importance)

# Print feature importance scores
for feature, importance in zip(X.columns, feature_importance):
    print(f"{feature}: {importance}")

# %%

# Decision trees:

dtree = tree.DecisionTreeClassifier(max_depth=3, random_state=1)
dtree.fit(X_train_scaled,y_train)
y_pred = dtree.predict(X_test_scaled)
y_pred_train1 = dtree.predict(X_train_scaled)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

#%%

# Variable importance:

feature_list = list(X.columns)
importances = list(dtree.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# Variable: LILATracts_1And20    Importance: 0.86
# Variable: LILATracts_halfAnd10 Importance: 0.06
# Variable: lasnap10             Importance: 0.05
# Variable: LowIncomeTracts      Importance: 0.02

#%%

#### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor
X = df_importance.drop('LILATracts_1And10', axis=1)
y = df_importance['LILATracts_1And10']
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("VIF Data: \n", vif_data)

# %%

#################
# Poverty rate
################

X = dff.drop('PovertyRate', axis=1)
y = dff['PovertyRate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
print("X_Train set shape: ", X_train.shape)
print("y_Train set shape: ", y_train.shape)
print("X_test set shape: ", X_test.shape)
print("y_test set shape: ", y_test.shape)

# %%

dtree = tree.DecisionTreeRegressor(max_depth=9, random_state=1)
dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)
y_pred_train1 = dtree.predict(X_train)

baseline_errors = abs(y_pred - y_test)
baseline_mape = 100 * np.mean((baseline_errors / y_test))
baseline_accuracy = 100 - baseline_mape

print('Metrics for d tree model for baseline')
print("Mean squared error: ", mean_squared_error(y_test, y_pred))
print('Average absolute error:', round(np.mean(baseline_errors), 2))

print('mean absolute percentage error (MAPE):', baseline_mape)
print('Accuracy:', round(baseline_accuracy, 2), '%.')
print("Train R2 score: ", r2_score(y_train,y_pred_train1))
print("Test R2 score: ", r2_score(y_test,y_pred))

#%%

# Variable importance:

feature_list = list(X.columns)
importances = list(dtree.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

# Variable: LowIncomeTracts      Importance: 0.58
# Variable: MedianFamilyIncome   Importance: 0.29
# Variable: lalowihalfshare      Importance: 0.04
# Variable: TractLOWI            Importance: 0.02
# Variable: PCTGQTRS             Importance: 0.01
# Variable: lapophalf            Importance: 0.01
# Variable: lakidshalfshare      Importance: 0.01
# Variable: TractKids            Importance: 0.01

# %%

from statsmodels.formula.api import ols
model = ols(formula='PovertyRate ~ LowIncomeTracts + MedianFamilyIncome + lalowihalfshare + PCTGQTRS + TractLOWI + lapophalf + lakidshalfshare + TractKids', data=dff)
model = model.fit()
print( model.summary() )

# %%
