#This scrpt will use a forward stepwise feature selection procedure
#Chandler Ross
#2/4/2022


#Import the modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SequentialFeatureSelector
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
df_train = pd.read_csv('C:/Users/eduroscha001/Documents/thesis/training_data/agg_td_30samp_even.csv')

#get the columns
print(df_train.columns, '\n', '\n')
row_len = len(df_train.index)
print('row #: ', row_len, '\n', '\n')

#clean the data by getting rid of unnecessary colums
#X & Y Data
# df_train.drop(['X.x'], axis=1, inplace=True)
# df_train.drop(['Y.x'], axis=1, inplace=True)
df_train.drop(['X.y'], axis=1, inplace=True)
df_train.drop(['Y.y'], axis=1, inplace=True)
#Reflectance Data
# df_train.drop(['SR_B1'], axis=1, inplace=True)
# df_train.drop(['SR_B2'], axis=1, inplace=True)
# df_train.drop(['SR_B3'], axis=1, inplace=True)
# df_train.drop(['SR_B4'], axis=1, inplace=True)
# df_train.drop(['SR_B5'], axis=1, inplace=True)
# df_train.drop(['SR_B6'], axis=1, inplace=True)
# df_train.drop(['SR_B7'], axis=1, inplace=True)
# df_train.drop(['ST_B6'], axis=1, inplace=True)
df_train.drop(['blue'], axis=1, inplace=True)
df_train.drop(['green'], axis=1, inplace=True)
df_train.drop(['red'], axis=1, inplace=True)
df_train.drop(['nir'], axis=1, inplace=True)
df_train.drop(['swir1'], axis=1, inplace=True)
df_train.drop(['swir2'], axis=1, inplace=True)
df_train.drop(['thermal'], axis=1, inplace=True)
#Raw burn threshold
# df_train.drop(['NoBurn'], axis=1, inplace=True)
# df_train.drop(['Burn'], axis=1, inplace=True)
df_train.drop(['VALUE_0'], axis=1, inplace=True)
df_train.drop(['VALUE_1'], axis=1, inplace=True)
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
# df_train.drop(['Samples_l7'], axis=1, inplace=True)
# df_train.drop(['Samples_l8'], axis=1, inplace=True)
# df_train.drop(['satellite'], axis=1, inplace=True)
df_train.drop(['sat'], axis=1, inplace=True)
# df_train.drop(['site.x'], axis=1, inplace=True)
# df_train.drop(['site.y'], axis=1, inplace=True)
# df_train.drop(['Samples_l5'], axis=1, inplace=True)
# df_train.drop(['ï..OID'], axis=1, inplace=True)
# df_train.drop(['ï..X'], axis=1, inplace=True)
# df_train.drop(['site.1'], axis=1, inplace=True)
df_train.drop(['site'], axis=1, inplace=True)
# df_train.drop(['row_id'], axis=1, inplace=True)
#Lag Reflectance
df_train.drop(['blue_1'], axis=1, inplace=True)
df_train.drop(['green_1'], axis=1, inplace=True)
df_train.drop(['red_1'], axis=1, inplace=True)
df_train.drop(['nir_1'], axis=1, inplace=True)
df_train.drop(['swir1_1'], axis=1, inplace=True)
df_train.drop(['thermal_1'], axis=1, inplace=True)
df_train.drop(['swir2_1'], axis=1, inplace=True)
df_train.drop(['blue_2'], axis=1, inplace=True)
df_train.drop(['green_2'], axis=1, inplace=True)
df_train.drop(['red_2'], axis=1, inplace=True)
df_train.drop(['nir_2'], axis=1, inplace=True)
df_train.drop(['swir1_2'], axis=1, inplace=True)
df_train.drop(['thermal_2'], axis=1, inplace=True)
df_train.drop(['swir2_2'], axis=1, inplace=True)
df_train.drop(['blue_3'], axis=1, inplace=True)
df_train.drop(['green_3'], axis=1, inplace=True)
df_train.drop(['red_3'], axis=1, inplace=True)
df_train.drop(['nir_3'], axis=1, inplace=True)
df_train.drop(['swir1_3'], axis=1, inplace=True)
df_train.drop(['thermal_3'], axis=1, inplace=True)
df_train.drop(['swir2_3'], axis=1, inplace=True)


#Drop the Variables that were bad
df_train.drop(['BAI'], axis=1, inplace=True)
df_train.drop(['VI46'], axis=1, inplace=True)
df_train.drop(['MIRBI_AVG'], axis=1, inplace=True)
df_train.drop(['MIRBI_STD'], axis=1, inplace=True)
df_train.drop(['dBAI'], axis=1, inplace=True)
df_train.drop(['BAI_STD'], axis=1, inplace=True)
df_train.drop(['NDWI_STD'], axis=1, inplace=True)
df_train.drop(['BAI_AVG'], axis=1, inplace=True)
df_train.drop(['EVI_AVG'], axis=1, inplace=True)
df_train.drop(['EVI_STD'], axis=1, inplace=True)
df_train.drop(['dEVI'], axis=1, inplace=True)


#The other burn Thresholds
df_train.drop(['X50Per'], axis=1, inplace=True) #***
df_train.drop(['X20Per'], axis=1, inplace=True) #***

#Check that only the Scene SVI, lag SVI, and Std SVI info is there
print('Columns to be Compared:', '\n', df_train.columns, '\n', '\n')

#Drops all rows with at least one null value. There shouldnt be any but we shall see.
df_train = df_train.dropna()
new_len = len(df_train.index)
amt_dropped = row_len - new_len
print("# of Rows Dropped: {}".format(amt_dropped))


#Export the CSV
# df_train.to_csv('C:/Users/eduroscha001/Documents/thesis/not_dropped_30_even.csv')
# print('End the script')

#Mark the time
step_0_t = time.time() - start_time
print('Step 0 took {} Seconds to complete. \n\n'.format(step_0_t))

#=======================================================================================================================
#                   Step 1: create independent and dependent variables
#=======================================================================================================================

# Then the dataframe is split into train and test datasets using sklean's train_test_split function
#Separate the dependent from the independent variables
var_columns = [c for c in df_train.columns if c not in ['X80Per']] #***

x = df_train.loc[:,var_columns] #predictors/independent variables
y = df_train.loc[:,'X80Per'] #dependent variable #***

#make the training and testing data from the data set
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.5, random_state=42)
print('x_train: {} \nx_valid: {} \ny_train: {}\ny_valid: {}\n'.format(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape))

#Mark the time
step_1_t = time.time() - start_time
print('Step 1 took {} Seconds to complete. \n\n'.format(step_1_t))

#=======================================================================================================================
#                   Step 2: Create a Simple GBM and Evaluate Performance
#=======================================================================================================================

#make the model
loss = 'deviance' # Default & Hawbaker used
learning_rate = 0.01 #Tune This
n_estimators = 3500 #Tune this
subsample = 0.5 # Hawbaker used (he also used 0.75)
criterion = 'friedman_mse' # Default
min_samples_split = 2 # Default
min_samples_leaf = 1 # Default
min_weight_fraction_leaf = 0.0 # Default
max_depth = 5 # Tune this
min_impurity_decrease = 0.0 # Default
init = None # Default
random_state = 25 # Hawbaker used
max_features = 'sqrt' # Hawbaker used
verbose = 0 # Default
max_leaf_nodes = None # Default
warm_start = False # Default
validation_fraction = 0.1 # Default
n_iter_no_change = None # Default
tol = 1e-4 # Default
ccp_alpha = 0.0 # Default

model_gbm = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                       subsample=subsample, criterion=criterion, min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth,
                                       min_impurity_decrease=min_impurity_decrease, init=init, random_state=random_state,
                                       max_features=max_features, verbose=verbose, max_leaf_nodes=max_leaf_nodes,
                                       warm_start=warm_start, validation_fraction=validation_fraction,
                                       n_iter_no_change=n_iter_no_change,tol=tol, ccp_alpha=ccp_alpha)

#train the model with the data
model_gbm.fit(x_train, y_train)


"""
List of predictors
['NDWI', 'VI45', 'NDVI', 'SAVI', 'VI43', 'NDMI', 'GEMI', 'NBR', 'CSI', 'NBR2_AVG', 'VI57', 'VI57_AVG',
                    'MIRBI', 'GEMI_AVG', 'NBR2', 'VI43_AVG', 'SAVI_AVG', 'VI6T_AVG', 'VI46_AVG', 'NDWI_AVG', 'NDVI_AVG',
                    'NBRT1_STD', 'VI45_AVG', 'NDMI_AVG', 'EVI', 'NBR_AVG', 'GEMI_STD', 'CSI_AVG', 'VI45_STD', 'NDMI_STD',
                    'NBRT1', 'VI46_STD', 'VI6T_STD', 'NDVI_STD', 'NBRT1_AVG', 'SAVI_STD', 'NBR_STD', 'NBR2_STD',
                    'VI57_STD', 'VI43_STD', 'CSI_STD', 'VI46', 'VI6T', 'BAI', 'EVI_STD', 'BAI_STD', 'EVI_AVG',
                    'BAI_AVG', 'MIRBI_AVG', 'MIRBI_STD', 'NDWI_STD', 'dBAI', 'dCSI', 'dEVI', 'dGEMI', 'dMIRBI',
                    'dNBR', 'dNBR2', 'dNBRT1', 'dNDMI', 'dNDVI', 'dNDWI', 'dSAVI', 'dVI6T', 'dVI43', 'dVI45', 'dVI46',
                    'dVI57']
"""

#Mark the time
step_2_t = time.time() - start_time
print('Step 2 took {} Seconds to complete. \n\n'.format(step_2_t))


#=======================================================================================================================
#                   Step 2.5: Feature Importance
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
step_3_t = time.time() - start_time
print('Step 3 took {} Seconds to complete. \n\n'.format(step_3_t))


#=======================================================================================================================
#                   Step 3: Forward Feature Selection
#=======================================================================================================================
'''
forward_feature_selector = SequentialFeatureSelector(model_gbm, n_features_to_select=15, direction='forward',
scoring='jaccard', cv=5, n_jobs=-1)
#fit the feature selector
forward_feature_selector.fit(x_train, y_train)
forward_feature_selector.get_params()
forward_feature_selector.get_support()
#Mark the time
step_3_t = time.time() - start_time
print('Step 3 took {} Seconds to complete. \n\n'.format(step_3_t))
'''
#=======================================================================================================================
#                   Step 3.1: Hawbaker Method of feature selection
#=======================================================================================================================
#Method didn't work
"""
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool
n_threads = 1
#List or predictors
predictors = ['NDWI', 'VI45', 'NDVI', 'SAVI', 'VI43', 'NDMI', 'GEMI', 'NBR', 'CSI', 'NBR2_AVG', 'VI57', 'VI57_AVG',
                'MIRBI', 'GEMI_AVG', 'NBR2', 'VI43_AVG', 'SAVI_AVG', 'VI6T_AVG', 'VI46_AVG', 'NDWI_AVG', 'NDVI_AVG',
                'NBRT1_STD', 'VI45_AVG', 'NDMI_AVG', 'EVI', 'NBR_AVG', 'GEMI_STD', 'CSI_AVG', 'VI45_STD', 'NDMI_STD',
                'NBRT1', 'VI46_STD', 'VI6T_STD', 'NDVI_STD', 'NBRT1_AVG', 'SAVI_STD', 'NBR_STD', 'NBR2_STD',
                'VI57_STD', 'VI43_STD', 'CSI_STD', 'VI46', 'VI6T', 'BAI', 'EVI_STD', 'BAI_STD', 'EVI_AVG',
                'BAI_AVG', 'MIRBI_AVG', 'MIRBI_STD', 'NDWI_STD', 'dBAI', 'dCSI', 'dEVI', 'dGEMI', 'dMIRBI',
                'dNBR', 'dNBR2', 'dNBRT1', 'dNDMI', 'dNDVI', 'dNDWI', 'dSAVI', 'dVI6T', 'dVI43', 'dVI45', 'dVI46',
                'dVI57']
#Stuff I am gathering from earlier in his code
auc = 0.5
# create the pool of workers
pool = ThreadPool(n_threads)#I have only 1 thread
prioritize_scene_predictors = True
predictors_in_model = []
def fitGBRM(temp_predictors, digits=8):
    # keep_predictors_mask = np.where(
    #     np.in1d(np.array(all_predictors_to_test), np.array(temp_predictors), invert=False))
    X_train2 = x_train#[:, keep_predictors_mask][:, 0, :]
    X_test2 = x_valid#[:, keep_predictors_mask][:, 0, :]
    clf = GradientBoostingClassifier(loss='deviance', learning_rate=learning_rate, n_estimators=n_estimators,
                                     max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state,
                                     subsample=subsample, max_features=max_features)
    clf.fit(X_train2, y_train)
    ##########
    # assess model fit using independent data
    ##########
    y_pred_proba_test = clf.predict_proba(X_test2)[:, 1]
    temp_auc = round(roc_auc_score(y_valid, y_pred_proba_test), digits)
    return (temp_auc)
#below is a multi thread thing, I need to figure out what to do about it
auc_list = pool.map(fitGBRM, [predictors_in_model + [test_predictor] for test_predictor in predictors])
# find the predictor that increased AUC the most
print("########################################")
if len(auc_list) == 0:
    pass
else:
    predictors_i = np.argsort(
        1.0 / np.array(auc_list)).tolist()  # use inverse of AUC so AUC values are sorted high to low
    ordered_predictors = np.array(predictors)[predictors_i]
    ordered_auc_list = np.array(auc_list)[predictors_i]
    # print the new AUC values for each test predictor
    for i in range(0, len(ordered_auc_list)):
        print(ordered_predictors[i], ordered_auc_list[i])
    if ordered_auc_list[0] > auc:
        test_predictor = ordered_predictors[0]
        predictors_in_model.append(test_predictor)  # add to predictors in model
        predictors.remove(test_predictor)  # remove from predictors to test
        # auc = round(ordered_auc_list[0],4)
        auc = ordered_auc_list[0]
        print("Adding", ordered_predictors[0], "to model predictor list", auc)
        # remove predictors that decreased AUC
        if len(predictors_in_model) > 1:
            predictors_to_remove = ordered_predictors[ordered_auc_list <= (last_auc - auc_threshold)]
            # predictors_to_remove = ordered_predictors[ordered_auc_list <= last_auc]
            print("These predictors were not helpful", predictors_to_remove, "remove them!")
            for prd in predictors_to_remove:
                if prd != 'random':
                    predictors.remove(prd)
print("########################################")
print("########################################")
print("Final model", predictors_in_model, "AUC:", auc)
print('End time = ' + str(datetime.now()))
endTime0 = time.time()
print('Processing time = ' + str(endTime0 - start_time) + ' seconds')
print('################################################################################')
"""