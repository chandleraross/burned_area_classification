#pickle maker saves the model to a pickle for later use
#Chandler Ross
#1/27/2022 - created the script
#5/03/2022 - Actually started on the script

#=======================================================================================================================


#Import the necessary libraries
#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.metrics import roc_auc_score
#import matplotlib.pylab as plt
import time, pickle
try:
    from osgeo import gdal
    from osgeo import gdalconst
except ImportError as e:
    print(e)
    import gdal
    import gdalconst

#I want to see how long the process takes
start_time = time.time()

#=======================================================================================================================
#                   Step 0: Read In the data and clean
#=======================================================================================================================
#import the data
df_train = pd.read_csv('../training_data/td20per.csv')


#get the columns
# print(df_train.columns)
row_len = len(df_train.index)
print('row #: ', row_len)

#clean the data by getting rid of unnecessary columns
df_train.drop(['X'], axis=1, inplace=True)
df_train.drop(['Y'], axis=1, inplace=True)
df_train.drop(['SR_B1'], axis=1, inplace=True)
df_train.drop(['SR_B2'], axis=1, inplace=True)
df_train.drop(['SR_B3'], axis=1, inplace=True)
df_train.drop(['SR_B4'], axis=1, inplace=True)
df_train.drop(['SR_B5'], axis=1, inplace=True)
df_train.drop(['SR_B6'], axis=1, inplace=True)
df_train.drop(['SR_B7'], axis=1, inplace=True)
df_train.drop(['NoBurn'], axis=1, inplace=True)
df_train.drop(['Burn'], axis=1, inplace=True)
df_train.drop(['ST_B6'], axis=1, inplace=True)
df_train.drop(['Samples_l7'], axis=1, inplace=True)
df_train.drop(['Samples_l8'], axis=1, inplace=True)
df_train.drop(['satellite'], axis=1, inplace=True)

# df_train.drop(['site'], axis=1, inplace=True)
# df_train.drop(['row_id'], axis=1, inplace=True)
# df_train.drop(['sat'], axis=1, inplace=True)
# df_train.drop(['X.x'], axis=1, inplace=True)
# df_train.drop(['Y.x'], axis=1, inplace=True)
# df_train.drop(['X.y'], axis=1, inplace=True)
# df_train.drop(['Y.y'], axis=1, inplace=True)
df_train.drop(['blue'], axis=1, inplace=True)
df_train.drop(['green'], axis=1, inplace=True)
df_train.drop(['red'], axis=1, inplace=True)
df_train.drop(['nir'], axis=1, inplace=True)
df_train.drop(['swir1'], axis=1, inplace=True)
df_train.drop(['swir2'], axis=1, inplace=True)
df_train.drop(['thermal'], axis=1, inplace=True)
# df_train.drop(['VALUE_0'], axis=1, inplace=True)
# df_train.drop(['VALUE_1'], axis=1, inplace=True)

#Drop the non important SVIs
df_train.drop(['BAI'], axis=1, inplace=True)
df_train.drop(['CSI'], axis=1, inplace=True)
df_train.drop(['EVI'], axis=1, inplace=True)
# df_train.drop(['GEMI'], axis=1, inplace=True)
df_train.drop(['MIRBI'], axis=1, inplace=True)
df_train.drop(['NBR'], axis=1, inplace=True)
df_train.drop(['NBR2'], axis=1, inplace=True)
df_train.drop(['NBRT1'], axis=1, inplace=True)
df_train.drop(['NDMI'], axis=1, inplace=True)
df_train.drop(['NDVI'], axis=1, inplace=True)
# df_train.drop(['NDWI'], axis=1, inplace=True)
df_train.drop(['SAVI'], axis=1, inplace=True)
df_train.drop(['VI6T'], axis=1, inplace=True)
df_train.drop(['VI43'], axis=1, inplace=True)
df_train.drop(['VI45'], axis=1, inplace=True)
df_train.drop(['VI46'], axis=1, inplace=True)
df_train.drop(['VI57'], axis=1, inplace=True)
df_train.drop(['dNBRT1'], axis=1, inplace=True)
df_train.drop(['dNDMI'], axis=1, inplace=True)
df_train.drop(['dNDVI'], axis=1, inplace=True)
df_train.drop(['dNDWI'], axis=1, inplace=True)
df_train.drop(['dSAVI'], axis=1, inplace=True)
df_train.drop(['dVI6T'], axis=1, inplace=True)
df_train.drop(['dVI45'], axis=1, inplace=True)
df_train.drop(['dVI46'], axis=1, inplace=True)
# df_train.drop(['dVI57'], axis=1, inplace=True)
df_train.drop(['dVI43'], axis=1, inplace=True)
# df_train.drop(['dNBR2'], axis=1, inplace=True)
df_train.drop(['dNBR'], axis=1, inplace=True)
df_train.drop(['dMIRBI'], axis=1, inplace=True)
df_train.drop(['dGEMI'], axis=1, inplace=True)
df_train.drop(['dEVI'], axis=1, inplace=True)
df_train.drop(['dCSI'], axis=1, inplace=True)
df_train.drop(['dBAI'], axis=1, inplace=True)
df_train.drop(['VI57_STD'], axis=1, inplace=True)
# df_train.drop(['VI46_STD'], axis=1, inplace=True)
df_train.drop(['VI45_STD'], axis=1, inplace=True)
df_train.drop(['VI43_STD'], axis=1, inplace=True)
df_train.drop(['VI6T_STD'], axis=1, inplace=True)
df_train.drop(['SAVI_STD'], axis=1, inplace=True)
df_train.drop(['NDWI_STD'], axis=1, inplace=True)
df_train.drop(['NDVI_STD'], axis=1, inplace=True)
df_train.drop(['NDMI_STD'], axis=1, inplace=True)
df_train.drop(['NBRT1_STD'], axis=1, inplace=True)
df_train.drop(['NBR2_STD'], axis=1, inplace=True)
df_train.drop(['NBR_STD'], axis=1, inplace=True)
df_train.drop(['MIRBI_STD'], axis=1, inplace=True)
df_train.drop(['GEMI_STD'], axis=1, inplace=True)
df_train.drop(['EVI_STD'], axis=1, inplace=True)
df_train.drop(['CSI_STD'], axis=1, inplace=True)
df_train.drop(['BAI_STD'], axis=1, inplace=True)
df_train.drop(['VI57_AVG'], axis=1, inplace=True)
# df_train.drop(['VI46_AVG'], axis=1, inplace=True)
df_train.drop(['VI45_AVG'], axis=1, inplace=True)
df_train.drop(['VI43_AVG'], axis=1, inplace=True)
df_train.drop(['VI6T_AVG'], axis=1, inplace=True)
df_train.drop(['SAVI_AVG'], axis=1, inplace=True)
df_train.drop(['NDWI_AVG'], axis=1, inplace=True)
df_train.drop(['NDVI_AVG'], axis=1, inplace=True)
# df_train.drop(['NDMI_AVG'], axis=1, inplace=True)
df_train.drop(['NBRT1_AVG'], axis=1, inplace=True)
df_train.drop(['NBR2_AVG'], axis=1, inplace=True)
df_train.drop(['NBR_AVG'], axis=1, inplace=True)
df_train.drop(['MIRBI_AVG'], axis=1, inplace=True)
# df_train.drop(['GEMI_AVG'], axis=1, inplace=True)
df_train.drop(['EVI_AVG'], axis=1, inplace=True)
df_train.drop(['CSI_AVG'], axis=1, inplace=True)
df_train.drop(['BAI_AVG'], axis=1, inplace=True)

#Raw averages and std values drop
df_train.drop(['swir2_avg'], axis=1, inplace=True)
df_train.drop(['thermal_avg'], axis=1, inplace=True)
df_train.drop(['swir1_avg'], axis=1, inplace=True)
df_train.drop(['nir_avg'], axis=1, inplace=True)
df_train.drop(['red_avg'], axis=1, inplace=True)
df_train.drop(['green_avg'], axis=1, inplace=True)
df_train.drop(['blue_avg'], axis=1, inplace=True)
df_train.drop(['swir2_std'], axis=1, inplace=True)
df_train.drop(['thermal_std'], axis=1, inplace=True)
df_train.drop(['swir1_std'], axis=1, inplace=True)
df_train.drop(['nir_std'], axis=1, inplace=True)
df_train.drop(['red_std'], axis=1, inplace=True)
df_train.drop(['green_std'], axis=1, inplace=True)
df_train.drop(['blue_std'], axis=1, inplace=True)
# df_train.drop(['swir2_3'], axis=1, inplace=True)
# df_train.drop(['thermal_3'], axis=1, inplace=True)
# df_train.drop(['swir1_3'], axis=1, inplace=True)
# df_train.drop(['nir_3'], axis=1, inplace=True)
# df_train.drop(['red_3'], axis=1, inplace=True)
# df_train.drop(['green_3'], axis=1, inplace=True)
# df_train.drop(['blue_3'], axis=1, inplace=True)
# df_train.drop(['swir2_2'], axis=1, inplace=True)
# df_train.drop(['thermal_2'], axis=1, inplace=True)
# df_train.drop(['swir1_2'], axis=1, inplace=True)
# df_train.drop(['nir_2'], axis=1, inplace=True)
# df_train.drop(['red_2'], axis=1, inplace=True)
# df_train.drop(['green_2'], axis=1, inplace=True)
# df_train.drop(['blue_2'], axis=1, inplace=True)
# df_train.drop(['swir2_1'], axis=1, inplace=True)
# df_train.drop(['thermal_1'], axis=1, inplace=True)
# df_train.drop(['swir1_1'], axis=1, inplace=True)
# df_train.drop(['nir_1'], axis=1, inplace=True)
# df_train.drop(['red_1'], axis=1, inplace=True)
# df_train.drop(['green_1'], axis=1, inplace=True)
# df_train.drop(['blue_1'], axis=1, inplace=True)


#make the ones for the different levels
df_train.drop(['X50Per'], axis=1, inplace=True)
df_train.drop(['X80Per'], axis=1, inplace=True)

# df_train.drop(['X20Per'], axis=1, inplace=True)
# df_train.drop(['X80Per'], axis=1, inplace=True)

# df_train.drop(['X50Per'], axis=1, inplace=True)
# df_train.drop(['X20Per'], axis=1, inplace=True)



#Drops all rows with at least one null value.
df_train = df_train.dropna()
new_len = len(df_train.index)
amt_dropped = row_len - new_len
print("# of Rows Dropped: {}".format(amt_dropped))

print(df_train.columns)

#=======================================================================================================================
#                   Step 1: create independent and dependent variables
#=======================================================================================================================

# Then the dataframe is split into train and test datasets using sklean's train_test_split function
#Separate the dependent from the independent variables
var_columns = [c for c in df_train.columns if c not in ['X20Per']]

x = df_train.loc[:,var_columns] #predictors/independent variables
y = df_train.loc[:,'X20Per'] #dependent variable


#make the training and testing data from the data set
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.5, random_state=42)
print('x_train: {} \nx_valid: {} \ny_train: {}\ny_valid: {}\n'.format(x_train.shape, x_valid.shape, y_train.shape, y_valid.shape))


#=======================================================================================================================
#                   Step 2: Create a Simple GBM and Evaluate Performance
#=======================================================================================================================


#make the model
model_gbm = GradientBoostingClassifier(loss = 'deviance', # Default & Hawbaker used
                           learning_rate = 0.05, # *Tune
                           n_estimators = 2500, # *Tune
                           subsample = 0.5, # Hawbaker used (he also used 0.75)
                           criterion = 'friedman_mse', # Default
                           min_samples_split = 2, # Default
                           min_samples_leaf = 1, # Default
                           min_weight_fraction_leaf = 0.0, # Default
                           max_depth = 5, # *Tune
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


with open('./models/stack_b_thresh_80.pkl', 'wb') as model_file:
    pickle.dump(model_gbm, model_file)


'''
Inputs
20% sampled data
20% Threshold
learning_rate=0.05
max_depth=5
n_estimators=2500
50% Threshold
learning_rate=0.01
max_depth=5
n_estimators=4000
80% Threshold
learning_rate=0.01
max_depth=5
n_estimators=3500
Models to make
stack_a_thresh_20.pkl
stack_a_thresh_50.pkl
stack_a_thresh_80.pkl
stack_b_thresh_20.pkl
stack_b_thresh_50.pkl
stack_b_thresh_80.pkl
'''