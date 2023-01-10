"""
Chandler Ross
Second Gradient Boosting Script
The goal of this script is to improve upon the one made before and to ...
"""
"""
Pseudo Code:
    Import libraries
 0: Import the data and get it ready 
 1: 
"""

#Import the necessary libraries
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

#=======================================================================================================================
#                   Step 0: Read In the data and clean
#=======================================================================================================================
#import the data
# df_train = pd.read_csv('E:/Thesis/Scripts/python_project/training_data/l5_7_training.csv')

#dummy data to ensure it works
df_train = pd.read_csv('F:/Thesis/Scripts/pirateShip/training_combined/miniSample2.csv')

#get the columns
print(df_train.columns)
row_len = len(df_train.index)
print('row #: ', row_len)

#clean the data by getting rid of unnecessary colums
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

#make the ones for the different levels
# df_train.drop(['X50Per'], axis=1, inplace=True)
# df_train.drop(['X80Per'], axis=1, inplace=True)

df_train.drop(['X50Per'], axis=1, inplace=True)
df_train.drop(['X80Per'], axis=1, inplace=True)

# df_train.drop(['X50Per'], axis=1, inplace=True)
# df_train.drop(['X20Per'], axis=1, inplace=True)


#Drops all rows with at least one null value.
df_train = df_train.dropna()
new_len = len(df_train.index)
amt_dropped = row_len - new_len
print("# of Rows Dropped: {}".format(amt_dropped))

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

#---------

#make the parameters
    #Learning Rate
    #In the paper her used these following learning rates
        #
        #
        #
    #Num estimators
    #In the paper her used these following estimators
        #
        #
        #
    #Max Depth
    #In the paper he  used the following tree depths
        #
        #
        #
    #min samples leaf
    #In the paper he  used the following tree depths
        #
        #
        #

#---------


#make the model
model_gbm = GradientBoostingClassifier(loss = 'deviance', # Default & Hawbaker used
                           learning_rate = 0.05,
                           n_estimators = 2500,
                           subsample = 0.5, # Hawbaker used (he also used 0.75)
                           criterion = 'friedman_mse', # Default
                           min_samples_split = 2, # Default
                           min_samples_leaf = 1, # Default
                           min_weight_fraction_leaf = 0.0, # Default
                           max_depth = 3, # Default
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

# with open('./model/ottos_pickle.pkl', 'wb') as model_file:
#     pickle.dump(model_gbm, model_file)
#
#
# with open('./model/ottos_pickle.pkl', 'rb') as model_file:
#     model_gbm = pickle.load(model_file)

# Look at how many estimators/trees were finally created during training
print("# of trees used in the model: ",len(model_gbm.estimators_))

#finds the performance on the training dataset and the validation training set
#gives the prediction of the probability
y_train_pred = model_gbm.predict_proba(x_train)[:,1]
y_valid_pred = model_gbm.predict_proba(x_valid)[:,1]

print("AUC Train: {:.4f}\nAUC Valid: {:.4f}".format(roc_auc_score(y_train, y_train_pred),
                                                    roc_auc_score(y_valid, y_valid_pred)))


#=======================================================================================================================
#                   Step 3: Look at Performance with Respect to Number of Trees
#=======================================================================================================================
"""
#staged_predict_proba function allows us to look at predictions at for different number of trees in the model
y_train_pred_trees = np.stack(list(model_gbm.staged_predict_proba(x_train)))[:,:,1]
y_valid_pred_trees = np.stack(list(model_gbm.staged_predict_proba(x_valid)))[:,:,1]
print(y_train_pred_trees.shape, y_valid_pred_trees.shape)
#shos how each additional tree changes the score
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
"""
#=======================================================================================================================
#                   Step 5: Apply the model to real Data
#=======================================================================================================================

'''
#call in the data
def read_data(fp_in_img, raster_driver_name='GTiff'):
    """
    register GDAL Driver and read input image file
    :param fp_in_img:
    :param raster_driver_name:
    :return: gdal.dataset
    """
    if raster_driver_name is None:
        gdal.AllRegister()
    else:
        driver = gdal.GetDriverByName(raster_driver_name)
        driver.Register()
    dataset = gdal.Open(fp_in_img, gdalconst.GA_ReadOnly)
    if dataset is None:
        print("Error: Could not read '{}'".format(fp_in_img))
        sys.exit()
    return dataset
raster_data = read_data('./output/LT05_CU_004013_20011012.tif')
'''
'''
#The following is from http://patrickgray.me/open-geo-tutorial/chapter_5_classification.html
import rasterio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.windows import Window
from rasterio.plot import reshape_as_raster, reshape_as_image
img_fp = './output/LE07_CU_004013_20000103.tif'
#change from rasterio to gdal numpy array, this should help things maybe
with rasterio.open(img_fp) as src:
    # may need to reduce this image size if your kernel crashes, takes a lot of memory
    img = src.read()[:, 150:600, 250:1400]
    #I will try the entire image
    # img = src.read()
# Take our full image and reshape into long 2d array (nrow * ncol, nband) for classification
# print(img.shape)
reshaped_img = reshape_as_image(img)
print(reshaped_img.shape)
new_arr = (img.shape[2] * img.shape[1], img.shape[0])
print(img[:17, :, :])
print(img.shape)
print(new_arr)
reshaped_img = img[:17, :, :].reshape(new_arr)
reshaped_img[np.isnan(reshaped_img)] = - 999
#check for NaNs
print(np.isnan(reshaped_img).any())
# #print(reshaped_img.reshape(-1, 17))
class_prediction = model_gbm.predict(reshaped_img.reshape(-1, 17))
# Reshape our classification map back into a 2D matrix so we can visualize it
# class_prediction = class_prediction.reshape(reshaped_img[0, :, :].shape)
class_prediction = class_prediction.reshape(img[0, :, :].shape)
def str_class_to_int(class_array):
    class_array[class_array == 'Not Burned'] = 0
    class_array[class_array == 'Burned'] = 1
    return(class_array.astype(int))
class_prediction = str_class_to_int(class_prediction)
print(class_prediction.shape)
num_rows = class_prediction.shape[0]
num_cols = class_prediction.shape[1]
import libs.indices as ind
ind.output_single_band_raster(data=class_prediction, out_fp='.gb_output/test.tif', col_size=num_cols,
                               row_size=num_rows, num_band=1, raster_driver_name='GTiff',
                               projection=None, geotransform=None, metadata=None, nodataval=-9999)
print(type(class_prediction))
'''

