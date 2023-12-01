
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

X = dff.drop('LILATracts_1And10', axis=1)
y = dff['LILATracts_1And10']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
print("X_Train set shape: ", X_train.shape)
print("y_Train set shape: ", y_train.shape)
print("X_test set shape: ", X_test.shape)
print("y_test set shape: ", y_test.shape)

# %%

dtree = tree.DecisionTreeClassifier(max_depth=3, random_state=1)
dtree.fit(X_train,y_train)
y_pred = dtree.predict(X_test)
y_pred_train1 = dtree.predict(X_train)

baseline_errors = abs(y_pred - y_test)
baseline_mape = 100 * np.mean((baseline_errors / y_test))
baseline_accuracy = 100 - baseline_mape

print('Metrics for Gradient boosting model for baseline')
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

# Variable: LILATracts_1And20    Importance: 0.86
# Variable: LILATracts_halfAnd10 Importance: 0.06
# Variable: lasnap10             Importance: 0.05
# Variable: LowIncomeTracts      Importance: 0.02

#%%

from statsmodels.formula.api import glm
import statsmodels.api as sm 

model = glm(formula='LILATracts_1And10 ~ C(LILATracts_1And20)', data=dff, family=sm.families.Binomial())
model_fit = model.fit()
print( model_fit.summary() )

# %%

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
