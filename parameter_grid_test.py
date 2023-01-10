#This Script aims to set which model parameter is the most accurate
#Chandler Ross
#1/27/2022

#=======================================================================================================================

#Import the modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import pickle

import matplotlib.pylab as plt
import os, time

#I want to see how long the process takes
start_time = time.time()

#***!!! Note
#I will need to change lines below when I change the burn threshold. Look for '#***' after the line
#83, 84, 105

#=======================================================================================================================
#                   Step 0: Read In the data and clean
#=======================================================================================================================
#import the data
#I will start with the 20% Sampled Data
df_train = pd.read_csv('F:/Thesis/Scripts/training_data/td20per.csv')

#get the columns
print(df_train.columns, '\n', '\n')
row_len = len(df_train.index)
print('row #: ', row_len, '\n', '\n')

#clean the data by getting rid of unnecessary colums
#X & Y Data
df_train.drop(['X'], axis=1, inplace=True)
df_train.drop(['Y'], axis=1, inplace=True)
#Reflectance Data
df_train.drop(['SR_B1'], axis=1, inplace=True)
df_train.drop(['SR_B2'], axis=1, inplace=True)
df_train.drop(['SR_B3'], axis=1, inplace=True)
df_train.drop(['SR_B4'], axis=1, inplace=True)
df_train.drop(['SR_B5'], axis=1, inplace=True)
df_train.drop(['SR_B6'], axis=1, inplace=True)
df_train.drop(['SR_B7'], axis=1, inplace=True)
df_train.drop(['ST_B6'], axis=1, inplace=True)
df_train.drop(['blue'], axis=1, inplace=True)
df_train.drop(['green'], axis=1, inplace=True)
df_train.drop(['red'], axis=1, inplace=True)
df_train.drop(['nir'], axis=1, inplace=True)
df_train.drop(['swir1'], axis=1, inplace=True)
df_train.drop(['swir2'], axis=1, inplace=True)
df_train.drop(['thermal'], axis=1, inplace=True)
#Raw burn threshold
df_train.drop(['NoBurn'], axis=1, inplace=True)
df_train.drop(['Burn'], axis=1, inplace=True)
#Lag Reflectance
df_train.drop(['blue_avg'], axis=1, inplace=True)
df_train.drop(['green_avg'], axis=1, inplace=True)
df_train.drop(['red_avg'], axis=1, inplace=True)
df_train.drop(['nir_avg'], axis=1, inplace=True)
df_train.drop(['swir1_avg'], axis=1, inplace=True)
df_train.drop(['thermal_avg'], axis=1, inplace=True)
df_train.drop(['swir2_avg'], axis=1, inplace=True)
#Lag Std
df_train.drop(['blue_std'], axis=1, inplace=True)
df_train.drop(['green_std'], axis=1, inplace=True)
df_train.drop(['red_std'], axis=1, inplace=True)
df_train.drop(['nir_std'], axis=1, inplace=True)
df_train.drop(['swir1_std'], axis=1, inplace=True)
df_train.drop(['thermal_std'], axis=1, inplace=True)
df_train.drop(['swir2_std'], axis=1, inplace=True)
#Misc
df_train.drop(['Samples_l7'], axis=1, inplace=True)
df_train.drop(['Samples_l8'], axis=1, inplace=True)
df_train.drop(['satellite'], axis=1, inplace=True)
#The other burn Thresholds
df_train.drop(['X50Per'], axis=1, inplace=True) #***
df_train.drop(['X80Per'], axis=1, inplace=True) #***

#Check that only the Scene SVI, lag SVI, and Std SVI info is there
print(df_train.columns, '\n', '\n')

#Drops all rows with at least one null value. There shouldnt be any but we shall see.
df_train = df_train.dropna()
new_len = len(df_train.index)
amt_dropped = row_len - new_len
print("# of Rows Dropped: {}".format(amt_dropped))

#Mark the time
step_0_t = time.time() - start_time
print('Step 0 took {} Seconds to complete. \n\n'.format(step_0_t))

#=======================================================================================================================
#                   Step 1: create independent and dependent variables
#=======================================================================================================================

# Then the dataframe is split into train and test datasets using sklean's train_test_split function
#Separate the dependent from the independent variables
var_columns = [c for c in df_train.columns if c not in ['X20Per']] #***

x = df_train.loc[:,var_columns] #predictors/independent variables
y = df_train.loc[:,'X20Per'] #dependent variable #***

#make the training and testing data from the data set
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.5, random_state=42)
print('x_train: {} \nx_valid: {} \ny_train: {}\ny_valid: {}\n'.format(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape))

#Mark the time
step_1_t = time.time() - start_time
print('Step 1 took {} Seconds to complete. \n\n'.format(step_1_t))

#=======================================================================================================================
#                   Step 2: Create a Simple GBM and Evaluate Performance
#=======================================================================================================================

#---------

#make the parameters
    #Learning Rate (learning_rate)
    #In the paper her used these following learning rates
        # 0.01
        # 0.05
        # 0.1

    #Number of Trees (n_estimators)
        # Least amount of trees needed

    #Splits per Tree (max_depth)
        # 1
        # 3
        # 5

#---------

#Make the grid
param_grid = {
    'learning_rate' : [0.01, 0.05, 0.1],
    'n_estimators' : [2000, 2500, 3000, 3500, 4000, 4500],
    'subsample' : [0.5],
    'max_depth' : [1, 3, 5],
    'random_state' : [25],
    'max_features' : ['sqrt']
}

#the classifier
gb = GradientBoostingClassifier()

#the grid search, may take a while
grid_search = GridSearchCV(estimator = gb, param_grid = param_grid, cv = 5, n_jobs = -1, werbose = 2)

# Fit the grid search to the data
grid_search.fit(train_x, train_y)

#Looks for the best params I guess
best_parameters = grid_search.best_params_
print(best_parameters)

#this may do the best params actually
rf_mod = grid_search.best_estimator_
print(rf_mod)
rfc_pred = rf_mod.predict(test_x)

#make below work after I finish

'''
#make the model when I know the best parameters
model_gbm = GradientBoostingClassifier(loss = 'deviance', # Default & Hawbaker used
                           learning_rate = 0.01, #Tune This
                           n_estimators = 4500, #Tune this
                           subsample = 0.5, # Hawbaker used (he also used 0.75)
                           criterion = 'friedman_mse', # Default
                           min_samples_split = 2, # Default
                           min_samples_leaf = 1, # Default
                           min_weight_fraction_leaf = 0.0, # Default
                           max_depth = 1, # Tune this
                           min_impurity_decrease = 0.0, # Default
                           init = None, # Default
                           random_state = 25, # Hawbaker used
                           max_features = 'sqrt', # Hawbaker used
                           verbose = 0, # Default
                           max_leaf_nodes = None, # Default
                           warm_start = False, # Default
                           validation_fraction = 0.1, # Default
                           n_iter_no_change = None, # Default
                           tol = 1e-4, # Default
                           ccp_alpha = 0.0 # Default
                           )
#train the model with the data
model_gbm.fit(x_train, y_train)
#Pickle Information, for when the right model is chosen
# with open('./model/ottos_pickle.pkl', 'wb') as model_file:
#     pickle.dump(model_gbm, model_file)
#
# with open('./model/ottos_pickle.pkl', 'rb') as model_file:
    model_gbm = pickle.load(model_file)
# Look at how many estimators/trees were finally created during training
print("# of trees used in the model: ",len(model_gbm.estimators_))
#finds the performance on the training dataset and the validation training set
#gives the prediction of the probability
y_train_pred = model_gbm.predict_proba(x_train)[:,1]
y_valid_pred = model_gbm.predict_proba(x_valid)[:,1]
print("AUC Train: {:.4f}\nAUC Valid: {:.4f}".format(roc_auc_score(y_train, y_train_pred),
                                                    roc_auc_score(y_valid, y_valid_pred)))
#Mark the time
step_2_t = time.time() - start_time
print('Step 2 took {} Seconds to complete. \n\n'.format(step_2_t))
#=======================================================================================================================
#                   Step 3: Look at Performance with Respect to Number of Trees
#=======================================================================================================================
#staged_predict_proba function allows us to look at predictions at for different number of trees in the model
y_train_pred_trees = np.stack(list(model_gbm.staged_predict_proba(x_train)))[:,:,1]
y_valid_pred_trees = np.stack(list(model_gbm.staged_predict_proba(x_valid)))[:,:,1]
print(y_train_pred_trees.shape, y_valid_pred_trees.shape)
#shows how each additional tree changes the score
auc_train_trees = [roc_auc_score(y_train, y_pred) for y_pred in y_train_pred_trees]
auc_valid_trees = [roc_auc_score(y_valid, y_pred) for y_pred in y_valid_pred_trees]
plt.figure(figsize=(12,5))
plt.plot(auc_train_trees, label='Train Data')
plt.plot(auc_valid_trees, label='Valid Data')
plt.title('AUC vs Number of Trees')
plt.ylabel('Area Under the Curve')
plt.xlabel('Number of Trees')
plt.legend()
#plot shows the model performance
print(plt.show())
#Mark the time
step_3_t = time.time() - start_time
print('Step 3 took {} Seconds to complete. \n\n'.format(step_3_t))
#=======================================================================================================================
#                   Step 4: Feature Importance
#=======================================================================================================================
#this shows how useful a predictor is
#Low importance features can be removed from the model for simpler, faster and more stable model
#get the columns of x
var_columns = x.columns
predictor_importance = pd.DataFrame({"Variable_Name":var_columns,
              "Importance":model_gbm.feature_importances_}) \
            .sort_values('Importance', ascending=False)
print(predictor_importance)
#Mark the time
step_4_t = time.time() - start_time
print('Step 4 took {} Seconds to complete. \n\n'.format(step_4_t))
'''