#lag_maker.py aims to make the SVI data from the lag data
#Chandler Ross
#1/27/2022

#=======================================================================================================================


#Import
from libs.indices import *
import rasterio, os, time, re
import libs.indices as ind
try:
    from osgeo import gdal
    from osgeo import gdalconst
except ImportError as e:
    print(e)
    import gdal
    import gdalconst
from math import sqrt
import numpy as np
from multiprocessing import Process, managers, cpu_count, current_process

#=======================================================================================================================
#                   Functions and Helpful Information
#=======================================================================================================================

#function to get the value for each of the corners
def corner_value(dataset, band_index, rows, cols, corner):
    """
    This provides a value for the corner to know which corner have values
    :param dataset: the raster dataset
    :param band: raster band
    :param cols: number of cols
    :param rows: number of rows
    :param corner: String, 'tl', 'tr', 'bl', 'br' This indicated which corner to get the data for
    :return: A value for the corner
    """

    # Get the specified band
    band = dataset.GetRasterBand(band_index)

    if corner == 'tl':
        data = band.ReadAsArray(0, 1087, 1, 1)
        return data
    if corner == 'tr':
        data = band.ReadAsArray(4909, 0, 1, 1)
        return data
    if corner == 'bl':
        data = band.ReadAsArray(1103, 5934, 1, 1)
        return data
    if corner == 'br':
        data = band.ReadAsArray(6005, 4848, 1, 1)
        return data
    else:
        print("Choose 'tl', 'tr', 'bl', or 'br'")



#Need to change this so that it works with regular expressions
# def find_file(file_name, file_path, corner1, corner2, band_index):
#     """
#     Returns a specific file if it meets the criteria
#     :param file_name: file name, string
#     :param file_path: file path, string
#     :param corner1: corner to match, string
#     :param corner2: corner to match, string
#     :param band_index: index of the desired band, integer
#     :return: a file that matches the name and the shape desired
#     """
#     # make the full path
#     #in_image = file_path + file_name
#
#     #Search for the image with regex
#     regex = re.compile(file_name)
#
#     for root, dirs, files in os.walk(file_path):
#         for file in files:
#             if regex.match(file):
#                 in_image = os.path.join(root, file)
#
#                 # make the dataset by reading in the data
#                 dataset = ind.read_data(in_image, raster_driver_name='GTiff')
#
#                 # Get the second raster band (Just a random band)
#                 #band = dataset.GetRasterBand(band_index)
#
#                 # get the dimensions of the image
#                 cols = dataset.RasterXSize
#                 rows = dataset.RasterYSize
#                 band = dataset.GetRasterBand(band_index)
#                 no_data_value = band.GetNoDataValue()
#
#
#                 c1 = corner_value(dataset, band_index, rows, cols, corner1)
#                 c2 = corner_value(dataset, band_index, rows, cols, corner2)
#
#                 if (np.isnan(c1[0,0]) and np.isnan(c2[0,0])):
#                     return None
#                 else:
#                     return in_image


#finds the file
def find_file(file_name, file_path, corner1, corner2, band_index, ls=8):
    """
    Returns a specific file if it meets the criteria
    :param file_name: file name, string
    :param file_path: file path, string
    :param corner1: corner to match, string
    :param corner2: corner to match, string
    :param band_index: index of the desired band, integer
    :return: a file that matches the name and the shape desired
    """
    # make the full path
    #in_image = file_path + file_name

    #Search for the image with regex
    regex = re.compile(file_name)

    #Empty list to place files that are in the same month as the target year
    match_list = []

    #For loop to append the possible files to test
    for root, dirs, files in os.walk(file_path):
        for file in files:
            if regex.match(file):
                in_image = os.path.join(root, file)
                match_list.append(in_image)

    #Test the different files and use the one that matters
    for file in match_list:

        # make the dataset by reading in the data
        dataset = ind.read_data(file, raster_driver_name='GTiff')

        # Get the second raster band (Just a random band)
        #band = dataset.GetRasterBand(band_index)

        # get the dimensions of the image
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        # band = dataset.GetRasterBand(band_index)
        # no_data_value = band.GetNoDataValue()


        c1 = corner_value(dataset, band_index, rows, cols, corner1)
        c2 = corner_value(dataset, band_index, rows, cols, corner2)

        if (ls ==8 or ls == 5):
            if (np.isnan(c1[0,0]) == False and np.isnan(c2[0,0]) == False):
                return file
        elif (ls == 7):
            if (np.isnan(c1[0,0]) == False or np.isnan(c2[0,0]) == False):
                return file

    # reset match_list to be empty (this may be unnecessary, im bad at coding)
    match_list = []



"""
#Band Index for the SVIs in the SVI stack
#  1  bai
#  2  csi
#  3  evi
#  4  gemi
#  5  miribi
#  6  nbr
#  7  nbr2
#  8  nbrt1
#  9  ndmi
#  10 ndvi
#  11 ndwi
#  12 savi
#  13 vi6t
#  14 vi43
#  15 vi45
#  16 vi46
#  17 vi57
Lag SVIs (6) that I will need for the input data with the input index
dNBR2       [7]
dVI57       [17]
NDMI_AVG    [9]
GEMI_AVG    [4]
VI46_AVG    [16]
VI6T_STD    [13]
# Sample naming convention for the SVI Stack
# LE07_CU_004013_20020124.tif
# LC08_CU_004013_20130324.tif
# LT05_CU_004013_20111001.tif
# 012345678901234567890123456
# 000000000011111111112222222
# index above
"""

'''
REGEX Example
https://stackoverflow.com/questions/39293968/how-do-i-search-directories-and-find-files-that-match-regex
import os
import re
rootdir = "/mnt/externa/Torrents/completed"
regex = re.compile('(.*zip$)|(.*rar$)|(.*r01$)')
for root, dirs, files in os.walk(rootdir):
  for file in files:
    if regex.match(file):
       print(file)
'''


#=======================================================================================================================
#                   Landsat 8
#=======================================================================================================================


#I am going to have to change this so that it incorporates different bands, or make it so that it just does a specific SVI

def l8_lag_svi_avg(in_path, out_path, band_index, band_name, MIN_YEAR_8, LAG_YEAR, operation='avg'):
    """
    Makes the lag average and dSVI for the images
    :param in_path: in data path, string, ex: './data/'
    :param out_path: out data path, string, ex: './output/'
    :param band_index: the index of the band of interest, integer
    :param band_name: the name of the SVI of interest, string
    :param operation: the name of the desired operation. Options: avg, std, or dif, string
    :return: Average, STDEV, and/or dSVI for specific SVIs
    """

    #helps with errors
    try:
    #something silly to nullify the try statement/ keep the index
    # if(1 == 1):

        # I want to see how long the process takes, this will mark the iterations
        iteration_num = 0

        files = os.listdir(in_path)

        #read through the files in the directory
        for file in os.listdir(in_path):
            if file.endswith('.tif'):

                #make the full path
                in_image = in_path + file

                #I want to see how long the process takes
                start_time = time.time()

                #make the dataset by reading in the data
                dataset = ind.read_data(in_image, raster_driver_name='GTiff')

                #get the projection
                projection = dataset.GetProjection()

                #get the geotransform
                geotransform = dataset.GetGeoTransform()

                #Get indexed raster band
                band = dataset.GetRasterBand(band_index)

                #get the no data value
                no_data_value = band.GetNoDataValue()

                #get the dimensions of the image
                cols = dataset.RasterXSize
                rows = dataset.RasterYSize


                #If statement for Landsat 8
                #checks if the top left and bottom left corners have data
                c1 = corner_value(dataset, band_index, rows, cols, 'tl')
                c2 = corner_value(dataset, band_index, rows, cols, 'bl')
                if (np.isnan(c1[0,0]) == False and np.isnan(c2[0,0]) == False):

                    #get the landsat name
                    out_sat = file[0:4]

                    #get the full input data for the output name
                    out_date = file[15:23]

                    #Get the date of the image
                    img_year = file[15:19]
                    #make it an integer
                    img_year_int = int(img_year)

                    #This line skips years that don't meet the criteria
                    if img_year_int < MIN_YEAR_8 + int(LAG_YEAR):
                        continue


                    #get the month
                    img_month = file[19:21]
                    img_month_int =int(img_month)
                    #get the days
                    img_day = file[21:23]
                    img_day_int = int(img_day)

                    #get the three year lag of the selected image
                    year_minus_1 = img_year_int - 1
                    year_minus_2 = img_year_int - 2
                    year_minus_3 = img_year_int - 3
                    #convert to string
                    yr_min_1_str = str(year_minus_1)
                    yr_min_2_str = str(year_minus_2)
                    yr_min_3_str = str(year_minus_3)

                    # LC08_CU_004013_20130324.tif
                    # expression pattern for landsat8
                    pattern1 = 'L..8.CU.004013.'
                    pattern2 = '...tif' #date and the .tif ending
                    pattern_full_yr1 = pattern1 + yr_min_1_str + img_month + pattern2
                    pattern_full_yr2 = pattern1 + yr_min_2_str + img_month + pattern2
                    pattern_full_yr3 = pattern1 + yr_min_3_str + img_month + pattern2
                    #* Above should work with regular expressions but it may need to be modified
                    # *** double check to see if the regular expressions work


                    #Test to see if the lag data exists for the month, if not get it from an adjacent month

                    #Lag 1
                    lag1 = find_file(file_name=pattern_full_yr1, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)
                    lag1_test = lag1
                    if lag1_test == None:
                        #check the day of the first image
                        if(img_day_int >= 15 and img_month == '12'):
                            #add a year and make january
                            yr_plus_1 = str(year_minus_1 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)

                        if(img_day_int >= 15 and img_month != '12'):
                            #just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)

                        if(img_day_int < 15 and img_month == '01'):
                            #subtract a year
                            #change month to december
                            year_minus_1_minus_1 = str(year_minus_1 - 1)

                            updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)

                        if(img_day_int < 15 and img_month != '01'):
                            #subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)

                    #Lag 2
                    lag2 = find_file(file_name=pattern_full_yr2, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)
                    lag2_test = lag2
                    if lag2_test == None:
                        #check the day of the first image
                        if(img_day_int >= 15 and img_month == '12'):
                            #add a year and make january
                            yr_plus_1 = str(year_minus_2 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)

                        if(img_day_int >= 15 and img_month != '12'):
                            #just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)

                        if(img_day_int < 15 and img_month == '01'):
                            #subtract a year
                            #change month to december
                            year_minus_2_minus_1 = str(year_minus_2 - 1)

                            updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)

                        if(img_day_int < 15 and img_month != '01'):
                            #subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)


                    #Lag 3
                    lag3 = find_file(file_name=pattern_full_yr3, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)
                    lag3_test = lag3
                    if lag3_test == None:
                        #check the day of the first image
                        if(img_day_int >= 15 and img_month == '12'):
                            #add a year and make january
                            yr_plus_1 = str(year_minus_3 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)

                        if(img_day_int >= 15 and img_month != '12'):
                            #just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)

                        if(img_day_int < 15 and img_month == '01'):
                            #subtract a year
                            #change month to december
                            year_minus_3_minus_1 = str(year_minus_3 - 1)

                            updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)

                        if(img_day_int < 15 and img_month != '01'):
                            #subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_3_str + updated_month + pattern2


                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index)

                    #make the dataset for the lags from the correct file paths
                    l1_ds = ind.read_data(lag1, raster_driver_name='GTiff')
                    l2_ds = ind.read_data(lag2, raster_driver_name='GTiff')
                    l3_ds = ind.read_data(lag3, raster_driver_name='GTiff')

                    #make the bands
                    l1_band = l1_ds.GetRasterBand(band_index)
                    l2_band = l2_ds.GetRasterBand(band_index)
                    l3_band = l3_ds.GetRasterBand(band_index)

                    #compute the operation
                    if(operation == 'avg'):
                        #Generate the avg lag data
                        operation_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                    elif(operation == 'std'):
                        # this is sqrt(sigma(value - mean) / number of instances )
                        #I am using sample standard deviation
                        #calculate the average
                        avgerage_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                        #calculate the StDev
                        operation_img = np.sqrt(
                            (
                                (l1_band.ReadAsArray() - avgerage_img) +
                                (l2_band.ReadAsArray() - avgerage_img) +
                                (l3_band.ReadAsArray() - avgerage_img)
                            ) / 2
                        )

                    elif(operation == 'dif'):
                        #get the lag average
                        avgerage_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                        #Call the origional image and make into a raster np array that can be manipulated
                        og_img = dataset.GetRasterBand(band_index)

                        #Calculate the differenced SVI (pre - post fire aka avg - og)
                        operation_img = avgerage_img - og_img.ReadAsArray()

                    else:
                        print("Pick either 'std', 'avg', or 'dif'")

                    #I am going to assume all the files have the same numbers of columns and rows

                    #Make the output filepath
                    # out_name = band_name + pattern1 + img_year + img_month + pattern2
                    # out_name = band_name + '_' + band_name
                    out_name = '{}_{}_{}.tif'.format(out_sat, band_name, out_date)

                    #Use OS path join
                    out_fp = os.path.join(out_path, out_name)
                    # out_fp = out_path + out_name



                    #Save the image to file
                    ind.output_single_band_raster(operation_img, out_fp, cols, rows, num_band=1,
                                                  raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                                                  metadata=None, nodataval=-9999)




                # checks if the top right and bottom right corners have data
                c1 = corner_value(dataset, band_index, rows, cols, 'tr')
                c2 = corner_value(dataset, band_index, rows, cols, 'br')
                if (np.isnan(c1[0,0]) == False and np.isnan(c2[0,0]) == False):

                    #get the landsat name
                    out_sat = file[0:4]

                    #get the full input data for the output name
                    out_date = file[15:23]

                    #Get the date of the image
                    img_year = file[15:19]
                    #make it an integer
                    img_year_int = int(img_year)

                    #This line skips years that don't meet the criteria
                    if img_year_int < MIN_YEAR_8 + int(LAG_YEAR):
                        continue


                    #get the month
                    img_month = file[19:21]
                    img_month_int =int(img_month)
                    #get the days
                    img_day = file[21:23]
                    img_day_int = int(img_day)

                    #get the three year lag of the selected image
                    year_minus_1 = img_year_int - 1
                    year_minus_2 = img_year_int - 2
                    year_minus_3 = img_year_int - 3
                    #convert to string
                    yr_min_1_str = str(year_minus_1)
                    yr_min_2_str = str(year_minus_2)
                    yr_min_3_str = str(year_minus_3)

                    # LC08_CU_004013_20130324.tif
                    # expression pattern for landsat8
                    pattern1 = 'L..8.CU.004013.'
                    pattern2 = '...tif' #date and the .tif ending
                    pattern_full_yr1 = pattern1 + yr_min_1_str + img_month + pattern2
                    pattern_full_yr2 = pattern1 + yr_min_2_str + img_month + pattern2
                    pattern_full_yr3 = pattern1 + yr_min_3_str + img_month + pattern2
                    #* Above should work with regular expressions but it may need to be modified
                    # *** double check to see if the regular expressions work


                    #Test to see if the lag data exists for the month, if not get it from an adjacent month

                    #Lag 1
                    lag1 = find_file(file_name=pattern_full_yr1, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)
                    lag1_test = lag1
                    if lag1_test == None:
                        #check the day of the first image
                        if(img_day_int >= 15 and img_month == '12'):
                            #add a year and make january
                            yr_plus_1 = str(year_minus_1 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)

                        if(img_day_int >= 15 and img_month != '12'):
                            #just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)

                        if(img_day_int < 15 and img_month == '01'):
                            #subtract a year
                            #change month to december
                            year_minus_1_minus_1 = str(year_minus_1 - 1)

                            updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)

                        if(img_day_int < 15 and img_month != '01'):
                            #subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)

                    #Lag 2
                    lag2 = find_file(file_name=pattern_full_yr2, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)
                    lag2_test = lag2
                    if lag2_test == None:
                        #check the day of the first image
                        if(img_day_int >= 15 and img_month == '12'):
                            #add a year and make january
                            yr_plus_1 = str(year_minus_2 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)

                        if(img_day_int >= 15 and img_month != '12'):
                            #just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)

                        if(img_day_int < 15 and img_month == '01'):
                            #subtract a year
                            #change month to december
                            year_minus_2_minus_1 = str(year_minus_2 - 1)

                            updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)

                        if(img_day_int < 15 and img_month != '01'):
                            #subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)


                    #Lag 3
                    lag3 = find_file(file_name=pattern_full_yr3, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)
                    lag3_test = lag3
                    if lag3_test == None:
                        #check the day of the first image
                        if(img_day_int >= 15 and img_month == '12'):
                            #add a year and make january
                            yr_plus_1 = str(year_minus_3 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)

                        if(img_day_int >= 15 and img_month != '12'):
                            #just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)

                        if(img_day_int < 15 and img_month == '01'):
                            #subtract a year
                            #change month to december
                            year_minus_3_minus_1 = str(year_minus_3 - 1)

                            updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)

                        if(img_day_int < 15 and img_month != '01'):
                            #subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_3_str + updated_month + pattern2


                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index)

                    #make the dataset for the lags from the correct file paths
                    l1_ds = ind.read_data(lag1, raster_driver_name='GTiff')
                    l2_ds = ind.read_data(lag2, raster_driver_name='GTiff')
                    l3_ds = ind.read_data(lag3, raster_driver_name='GTiff')

                    #make the bands
                    l1_band = l1_ds.GetRasterBand(band_index)
                    l2_band = l2_ds.GetRasterBand(band_index)
                    l3_band = l3_ds.GetRasterBand(band_index)

                    #compute the operation
                    if(operation == 'avg'):
                        #Generate the avg lag data
                        operation_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                    elif(operation == 'std'):
                        # this is sqrt(sigma(value - mean) / number of instances )
                        #I am using sample standard deviation
                        #calculate the average
                        avgerage_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                        #calculate the StDev
                        operation_img = np.sqrt(
                            (
                                (l1_band.ReadAsArray() - avgerage_img) +
                                (l2_band.ReadAsArray() - avgerage_img) +
                                (l3_band.ReadAsArray() - avgerage_img)
                            ) / 2
                        )

                    elif(operation == 'dif'):
                        #get the lag average
                        avgerage_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                        #Call the origional image and make into a raster np array that can be manipulated
                        og_img = dataset.GetRasterBand(band_index)

                        #Calculate the differenced SVI (pre - post fire aka avg - og)
                        operation_img = avgerage_img - og_img.ReadAsArray()

                    else:
                        print("Pick either 'std', 'avg', or 'dif'")

                    #I am going to assume all the files have the same numbers of columns and rows

                    #Make the output filepath
                    # out_name = band_name + pattern1 + img_year + img_month + pattern2
                    # out_name = band_name + '_' + band_name
                    out_name = '{}_{}_{}.tif'.format(out_sat, band_name, out_date)

                    # Use OS path join
                    out_fp = os.path.join(out_path, out_name)
                    # out_fp = out_path + out_name



                    # Save the image to file
                    ind.output_single_band_raster(operation_img, out_fp, cols, rows, num_band=1,
                                                  raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                                                  metadata=None, nodataval=-9999)



                # I want to see how long the process takes
                time_dif = time.time() - start_time
                iteration_num += 1
                print('Iteration {} took {} seconds to complete'.format(iteration_num, time_dif))

    except Exception as e:
        print(e)


#=======================================================================================================================
#                   Landsat 7
#=======================================================================================================================

def l7_lag_svi_avg(in_path, out_path, band_index, band_name, MIN_YEAR_7, LAG_YEAR, operation='avg'):
    """
    Makes the lag average and dSVI for the images
    :param in_path: in data path, string, ex: './data/'
    :param out_path: out data path, string, ex: './output/'
    :param band_index: the index of the band of interest, integer
    :param band_name: the name of the SVI of interest, string
    :param operation: the name of the desired operation. Options: avg, std, or dif, string
    :return: Average, STDEV, and/or dSVI for specific SVIs
    """

    #helps with errors
    # try:
    #something silly to nullify the try statement/ keep the index
    if(1 == 1):

        # I want to see how long the process takes, this will mark the iterations
        iteration_num = 0

        files = os.listdir(in_path)

        #read through the files in the directory
        for file in os.listdir(in_path):
            if file.endswith('.tif'):

                #make the full path
                in_image = in_path + file

                #I want to see how long the process takes
                start_time = time.time()

                #make the dataset by reading in the data
                dataset = ind.read_data(in_image, raster_driver_name='GTiff')

                #get the projection
                projection = dataset.GetProjection()

                #get the geotransform
                geotransform = dataset.GetGeoTransform()

                #Get indexed raster band
                band = dataset.GetRasterBand(band_index)

                #get the no data value
                no_data_value = band.GetNoDataValue()

                #get the dimensions of the image
                cols = dataset.RasterXSize
                rows = dataset.RasterYSize


                #If statement for Landsat 8
                #checks if the top left and bottom left corners have data
                c1 = corner_value(dataset, band_index, rows, cols, 'tl')
                c2 = corner_value(dataset, band_index, rows, cols, 'bl')
                if (np.isnan(c1[0,0]) == False or np.isnan(c2[0,0]) == False):

                    #get the landsat name
                    out_sat = file[0:4]

                    #get the full input data for the output name
                    out_date = file[15:23]

                    #Get the date of the image
                    img_year = file[15:19]
                    #make it an integer
                    img_year_int = int(img_year)

                    #This line skips years that don't meet the criteria
                    if img_year_int < MIN_YEAR_7 + int(LAG_YEAR):
                        continue


                    #get the month
                    img_month = file[19:21]
                    img_month_int =int(img_month)
                    #get the days
                    img_day = file[21:23]
                    img_day_int = int(img_day)

                    #get the three year lag of the selected image
                    year_minus_1 = img_year_int - 1
                    year_minus_2 = img_year_int - 2
                    year_minus_3 = img_year_int - 3
                    #convert to string
                    yr_min_1_str = str(year_minus_1)
                    yr_min_2_str = str(year_minus_2)
                    yr_min_3_str = str(year_minus_3)

                    # LC08_CU_004013_20130324.tif
                    # expression pattern for landsat8
                    pattern1 = 'L..7.CU.004013.'
                    pattern2 = '...tif' #date and the .tif ending
                    pattern_full_yr1 = pattern1 + yr_min_1_str + img_month + pattern2
                    pattern_full_yr2 = pattern1 + yr_min_2_str + img_month + pattern2
                    pattern_full_yr3 = pattern1 + yr_min_3_str + img_month + pattern2
                    #* Above should work with regular expressions but it may need to be modified
                    # *** double check to see if the regular expressions work


                    #Test to see if the lag data exists for the month, if not get it from an adjacent month

                    #Lag 1
                    lag1 = find_file(file_name=pattern_full_yr1, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)
                    lag1_test = lag1
                    if lag1_test == None:
                        #Test 1
                        #check the day of the first image
                        if(img_day_int >= 15 and img_month == '12'):
                            #add a year and make january
                            yr_plus_1 = str(year_minus_1 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)


                            #above didnt work so get the month before
                            if(lag1 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            #above didnt work so get 2 months after
                            if(lag1 == None):
                                # add a year and make Feburary
                                yr_plus_1 = str(year_minus_1 + 1)
                                updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            #above didnt work so get 2 months before
                            if(lag1 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                        #Test 2
                        if(img_day_int >= 15 and img_month != '12'):
                            #just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)


                            #above didnt work so get the month before
                            if(lag1 == None):
                                #check if the month is January
                                if(img_month == '01'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)


                            #above didnt work so get 2 months after
                            if(lag1 == None):
                                if(img_month == '11'):
                                    # add a year and make January
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                                #add two months
                                month_plus_2 = str(img_month_int + 1)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2

                                updated_name = pattern1 + str(year_minus_1) + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=7)

                            #above didnt work so get 2 months before
                            if(lag1 == None):
                                #check to see if the month is Jan or Feb
                                if(img_month == '01'):
                                    # subtract a year
                                    # change month to november
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '11' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                if(img_month == '02'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                    # subtract 2 months
                                    updated_month = str(img_month_int - 2)
                                    if int(updated_month) < 10:
                                        updated_month = '0' + updated_month
                                    updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                     band_index=band_index, ls=7)

                        #Test 3
                        if(img_day_int < 15 and img_month == '01'):
                            #subtract a year
                            #change month to december
                            year_minus_1_minus_1 = str(year_minus_1 - 1)

                            updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)


                            #above didnt work so add a month
                            if(lag1 == None):
                                #just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                                #rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)


                            #above didnt work so subtract 2 months
                            if(lag1 == None):
                                # subtract a year
                                # change month to november
                                year_minus_1_minus_1 = str(year_minus_1 - 1)

                                updated_name = pattern1 + year_minus_1_minus_1 + '11' + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            #above didnt work so add 2 months
                            if(lag1 == None):
                                #b/c its jan just add 2 months
                                month_plus_1 = str(img_month_int + 2)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                                #rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)


                        #Test 4
                        if(img_day_int < 15 and img_month != '01'):
                            #subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)



                            #above didnt work so get one month after
                            if(lag1 == None):
                                #check for december
                                if(img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                #Just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                                #rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)

                            # above didnt work so get 2 months before
                            if (lag1 == None):
                                #check for feburary
                                if(img_month == '02'):
                                    # change month to december
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                     band_index=band_index, ls=7)

                                #subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            #above didnt work so add 2 months
                            if(lag1 == None):
                                # check for november
                                if (img_month == '11'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                #check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                # Just add 2 months
                                month_plus_2 = str(img_month_int + 2)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2
                                updated_name = pattern1 + yr_min_1_str + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)


                    #Lag 2
                    lag2 = find_file(file_name=pattern_full_yr2, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)
                    lag2_test = lag2
                    if lag2_test == None:
                        #Test 1
                        #check the day of the first image
                        if(img_day_int >= 15 and img_month == '12'):
                            #add a year and make january
                            yr_plus_1 = str(year_minus_2 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)

                            #above didnt work so get the month before
                            if(lag2 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            #above didnt work so get 2 months after
                            if(lag2 == None):
                                # add a year and make Feburary
                                yr_plus_1 = str(year_minus_2 + 1)
                                updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            #above didnt work so get 2 months before
                            if(lag2 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)


                        #Test 2
                        if(img_day_int >= 15 and img_month != '12'):
                            #just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)

                            # above didnt work so get the month before
                            if (lag2 == None):
                                # check if the month is January
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            # above didnt work so get 2 months after
                            if (lag2 == None):
                                if (img_month == '11'):
                                    # add a year and make January
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl',
                                                     band_index=band_index, ls=7)

                                # add two months
                                month_plus_2 = str(img_month_int + 1)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2

                                updated_name = pattern1 + year_minus_2 + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            # above didnt work so get 2 months before
                            if (lag2 == None):
                                # check to see if the month is Jan or Feb
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to november
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '11' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                if (img_month == '02'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)



                        #Test 3
                        if(img_day_int < 15 and img_month == '01'):
                            #subtract a year
                            #change month to december
                            year_minus_2_minus_1 = str(year_minus_2 - 1)

                            updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)

                            # above didnt work so add a month
                            if (lag2 == None):
                                # just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            # above didnt work so subtract 2 months
                            if (lag2 == None):
                                # subtract a year
                                # change month to november
                                year_minus_2_minus_1 = str(year_minus_2 - 1)

                                updated_name = pattern1 + year_minus_2_minus_1 + '11' + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            # above didnt work so add 2 months
                            if (lag2 == None):
                                # b/c its jan just add 2 months
                                month_plus_1 = str(img_month_int + 2)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)


                        #Test 4
                        if(img_day_int < 15 and img_month != '01'):
                            #subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)

                            #above didnt work so get one month after
                            if(lag2 == None):
                                #check for december
                                if(img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                #Just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                                #rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)

                            # above didnt work so get 2 months before
                            if (lag2 == None):
                                #check for feburary
                                if(img_month == '02'):
                                    # change month to december
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                     band_index=band_index, ls=7)

                                #subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            #above didnt work so add 2 months
                            if(lag2 == None):
                                # check for november
                                if (img_month == '11'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                #check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                # Just add 2 months
                                month_plus_2 = str(img_month_int + 2)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2
                                updated_name = pattern1 + yr_min_2_str + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)


                    #Lag 3
                    lag3 = find_file(file_name=pattern_full_yr3, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)
                    lag3_test = lag3
                    if lag3_test == None:
                        #Test1
                        #check the day of the first image
                        if(img_day_int >= 15 and img_month == '12'):
                            #add a year and make january
                            yr_plus_1 = str(year_minus_3 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)

                            # above didnt work so get the month before
                            if (lag3 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                la3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            # above didnt work so get 2 months after
                            if (lag3 == None):
                                # add a year and make Feburary
                                yr_plus_1 = str(year_minus_3 + 1)
                                updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            # above didnt work so get 2 months before
                            if (lag3 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)


                        #Test 2
                        if(img_day_int >= 15 and img_month != '12'):
                            #just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)

                           # above didnt work so get the month before
                            if (lag3 == None):
                                # check if the month is January
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            # above didnt work so get 2 months after
                            if (lag3 == None):
                                if (img_month == '11'):
                                    # add a year and make January
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl',
                                                     band_index=band_index, ls=7)

                                # add two months
                                month_plus_2 = str(img_month_int + 1)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2

                                updated_name = pattern1 + str(year_minus_3) + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            # above didnt work so get 2 months before
                            if (lag3 == None):
                                # check to see if the month is Jan or Feb
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to november
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '11' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                if (img_month == '02'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)



                        #Test 3
                        if(img_day_int < 15 and img_month == '01'):
                            #subtract a year
                            #change month to december
                            year_minus_3_minus_1 = str(year_minus_3 - 1)

                            updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)

                            # above didnt work so add a month
                            if (lag3 == None):
                                # just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            # above didnt work so subtract 2 months
                            if (lag3 == None):
                                # subtract a year
                                # change month to november
                                year_minus_3_minus_1 = str(year_minus_3 - 1)

                                updated_name = pattern1 + year_minus_3_minus_1 + '11' + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            # above didnt work so add 2 months
                            if (lag3 == None):
                                # b/c its jan just add 2 months
                                month_plus_1 = str(img_month_int + 2)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)


                        #Test 4
                        if(img_day_int < 15 and img_month != '01'):
                            #subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_3_str + updated_month + pattern2


                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)

                            #above didnt work so get one month after
                            if(lag3 == None):
                                #check for december
                                if(img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                #Just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                                #rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl', band_index=band_index, ls=7)

                            # above didnt work so get 2 months before
                            if (lag3 == None):
                                #check for feburary
                                if(img_month == '02'):
                                    # change month to december
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                     band_index=band_index, ls=7)

                                #subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)

                            #above didnt work so add 2 months
                            if(lag3 == None):
                                # check for november
                                if (img_month == '11'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                #check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=7)

                                # Just add 2 months
                                month_plus_2 = str(img_month_int + 2)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2
                                updated_name = pattern1 + yr_min_3_str + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=7)


                    #code to trouble shoot the unexpected values
                    print('lag1: {}\nlag2: {}\nlag3: {}'.format(lag1, lag2, lag3))
                    #make the dataset for the lags from the correct file paths
                    l1_ds = ind.read_data(lag1, raster_driver_name='GTiff')
                    l2_ds = ind.read_data(lag2, raster_driver_name='GTiff')
                    l3_ds = ind.read_data(lag3, raster_driver_name='GTiff')

                    #make the bands
                    l1_band = l1_ds.GetRasterBand(band_index)
                    l2_band = l2_ds.GetRasterBand(band_index)
                    l3_band = l3_ds.GetRasterBand(band_index)

                    #compute the operation
                    if(operation == 'avg'):
                        #Generate the avg lag data
                        operation_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                    elif(operation == 'std'):
                        # this is sqrt(sigma(value - mean) / number of instances )
                        #I am using sample standard deviation
                        #calculate the average
                        avgerage_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                        #calculate the StDev
                        operation_img = np.sqrt(
                            (
                                (l1_band.ReadAsArray() - avgerage_img) +
                                (l2_band.ReadAsArray() - avgerage_img) +
                                (l3_band.ReadAsArray() - avgerage_img)
                            ) / 2
                        )

                    elif(operation == 'dif'):
                        #get the lag average
                        avgerage_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                        #Call the origional image and make into a raster np array that can be manipulated
                        og_img = dataset.GetRasterBand(band_index)

                        #Calculate the differenced SVI (pre - post fire aka avg - og)
                        operation_img = avgerage_img - og_img.ReadAsArray()

                    else:
                        print("Pick either 'std', 'avg', or 'dif'")

                    #I am going to assume all the files have the same numbers of columns and rows

                    #Make the output filepath
                    # out_name = band_name + pattern1 + img_year + img_month + pattern2
                    # out_name = band_name + '_' + band_name
                    out_name = '{}_{}_{}.tif'.format(out_sat, band_name, out_date)

                    #Use OS path join
                    out_fp = os.path.join(out_path, out_name)
                    # out_fp = out_path + out_name



                    #Save the image to file
                    ind.output_single_band_raster(operation_img, out_fp, cols, rows, num_band=1,
                                                  raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                                                  metadata=None, nodataval=-9999)




                # checks if the top right and bottom right corners have data
                c1 = corner_value(dataset, band_index, rows, cols, 'tr')
                c2 = corner_value(dataset, band_index, rows, cols, 'br')
                if (np.isnan(c1[0,0]) == False or np.isnan(c2[0,0]) == False):

                    #get the landsat name
                    out_sat = file[0:4]

                    #get the full input data for the output name
                    out_date = file[15:23]

                    #Get the date of the image
                    img_year = file[15:19]
                    #make it an integer
                    img_year_int = int(img_year)

                    #This line skips years that don't meet the criteria
                    if img_year_int < MIN_YEAR_7 + int(LAG_YEAR):
                        continue


                    #get the month
                    img_month = file[19:21]
                    img_month_int =int(img_month)
                    #get the days
                    img_day = file[21:23]
                    img_day_int = int(img_day)

                    #get the three year lag of the selected image
                    year_minus_1 = img_year_int - 1
                    year_minus_2 = img_year_int - 2
                    year_minus_3 = img_year_int - 3
                    #convert to string
                    yr_min_1_str = str(year_minus_1)
                    yr_min_2_str = str(year_minus_2)
                    yr_min_3_str = str(year_minus_3)

                    # LC08_CU_004013_20130324.tif
                    # expression pattern for landsat8
                    pattern1 = 'L..7.CU.004013.'
                    pattern2 = '...tif' #date and the .tif ending
                    pattern_full_yr1 = pattern1 + yr_min_1_str + img_month + pattern2
                    pattern_full_yr2 = pattern1 + yr_min_2_str + img_month + pattern2
                    pattern_full_yr3 = pattern1 + yr_min_3_str + img_month + pattern2
                    #* Above should work with regular expressions but it may need to be modified
                    # *** double check to see if the regular expressions work


                    #Test to see if the lag data exists for the month, if not get it from an adjacent month

                    #Lag 1
                    lag1 = find_file(file_name=pattern_full_yr1, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)
                    lag1_test = lag1
                    if lag1_test == None:
                        #Test 1
                        #check the day of the first image
                        if(img_day_int >= 15 and img_month == '12'):
                            #add a year and make january
                            yr_plus_1 = str(year_minus_1 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)


                            #above didnt work so get the month before
                            if(lag1 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            #above didnt work so get 2 months after
                            if(lag1 == None):
                                # add a year and make Feburary
                                yr_plus_1 = str(year_minus_1 + 1)
                                updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            #above didnt work so get 2 months before
                            if(lag1 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                        #Test 2
                        if(img_day_int >= 15 and img_month != '12'):
                            #just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)


                            #above didnt work so get the month before
                            if(lag1 == None):
                                #check if the month is January
                                if(img_month == '01'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)


                            #above didnt work so get 2 months after
                            if(lag1 == None):
                                if(img_month == '11'):
                                    # add a year and make January
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                                #add two months
                                month_plus_2 = str(img_month_int + 1)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2

                                updated_name = pattern1 + str(year_minus_1) + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=7)

                            #above didnt work so get 2 months before
                            if(lag1 == None):
                                #check to see if the month is Jan or Feb
                                if(img_month == '01'):
                                    # subtract a year
                                    # change month to november
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '11' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                if(img_month == '02'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                    # subtract 2 months
                                    updated_month = str(img_month_int - 2)
                                    if int(updated_month) < 10:
                                        updated_month = '0' + updated_month
                                    updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                     band_index=band_index, ls=7)

                        #Test 3
                        if(img_day_int < 15 and img_month == '01'):
                            #subtract a year
                            #change month to december
                            year_minus_1_minus_1 = str(year_minus_1 - 1)

                            updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)


                            #above didnt work so add a month
                            if(lag1 == None):
                                #just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                                #rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)


                            #above didnt work so subtract 2 months
                            if(lag1 == None):
                                # subtract a year
                                # change month to november
                                year_minus_1_minus_1 = str(year_minus_1 - 1)

                                updated_name = pattern1 + year_minus_1_minus_1 + '11' + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            #above didnt work so add 2 months
                            if(lag1 == None):
                                #b/c its jan just add 2 months
                                month_plus_1 = str(img_month_int + 2)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                                #rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                        #Test 4
                        if(img_day_int < 15 and img_month != '01'):
                            #subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                            #rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)



                            #above didnt work so get one month after
                            if(lag1 == None):
                                #check for december
                                if(img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                #Just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                                #rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                            # above didnt work so get 2 months before
                            if (lag1 == None):
                                #check for feburary
                                if(img_month == '02'):
                                    # change month to december
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                     band_index=band_index, ls=7)

                                #subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            #above didnt work so add 2 months
                            if(lag1 == None):
                                # check for november
                                if (img_month == '11'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                #check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                # Just add 2 months
                                month_plus_2 = str(img_month_int + 2)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2
                                updated_name = pattern1 + yr_min_1_str + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)


                    #Lag 2
                    lag2 = find_file(file_name=pattern_full_yr2, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)
                    lag2_test = lag2
                    if lag2_test == None:
                        #Test 1
                        #check the day of the first image
                        if(img_day_int >= 15 and img_month == '12'):
                            #add a year and make january
                            yr_plus_1 = str(year_minus_2 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                            #above didnt work so get the month before
                            if(lag2 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            #above didnt work so get 2 months after
                            if(lag2 == None):
                                # add a year and make Feburary
                                yr_plus_1 = str(year_minus_2 + 1)
                                updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            #above didnt work so get 2 months before
                            if(lag2 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)


                        #Test 2
                        if(img_day_int >= 15 and img_month != '12'):
                            #just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                            # above didnt work so get the month before
                            if (lag2 == None):
                                # check if the month is January
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            # above didnt work so get 2 months after
                            if (lag2 == None):
                                if (img_month == '11'):
                                    # add a year and make January
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br',
                                                     band_index=band_index, ls=7)

                                # add two months
                                month_plus_2 = str(img_month_int + 1)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2

                                updated_name = pattern1 + str(year_minus_2) + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            # above didnt work so get 2 months before
                            if (lag2 == None):
                                # check to see if the month is Jan or Feb
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to november
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '11' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                if (img_month == '02'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)



                        #Test 3
                        if(img_day_int < 15 and img_month == '01'):
                            #subtract a year
                            #change month to december
                            year_minus_2_minus_1 = str(year_minus_2 - 1)

                            updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                            # above didnt work so add a month
                            if (lag2 == None):
                                # just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            # above didnt work so subtract 2 months
                            if (lag2 == None):
                                # subtract a year
                                # change month to november
                                year_minus_2_minus_1 = str(year_minus_2 - 1)

                                updated_name = pattern1 + year_minus_2_minus_1 + '11' + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            # above didnt work so add 2 months
                            if (lag2 == None):
                                # b/c its jan just add 2 months
                                month_plus_1 = str(img_month_int + 2)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)


                        #Test 4
                        if(img_day_int < 15 and img_month != '01'):
                            #subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                            #rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                            #above didnt work so get one month after
                            if(lag2 == None):
                                #check for december
                                if(img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                #Just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                                #rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                            # above didnt work so get 2 months before
                            if (lag2 == None):
                                #check for feburary
                                if(img_month == '02'):
                                    # change month to december
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                     band_index=band_index, ls=7)

                                #subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            #above didnt work so add 2 months
                            if(lag2 == None):
                                # check for november
                                if (img_month == '11'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                #check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                # Just add 2 months
                                month_plus_2 = str(img_month_int + 2)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2
                                updated_name = pattern1 + yr_min_2_str + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)


                    #Lag 3
                    lag3 = find_file(file_name=pattern_full_yr3, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)
                    lag3_test = lag3
                    if lag3_test == None:
                        #Test1
                        #check the day of the first image
                        if(img_day_int >= 15 and img_month == '12'):
                            #add a year and make january
                            yr_plus_1 = str(year_minus_3 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                            # above didnt work so get the month before
                            if (lag3 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                la3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            # above didnt work so get 2 months after
                            if (lag3 == None):
                                # add a year and make Feburary
                                yr_plus_1 = str(year_minus_3 + 1)
                                updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            # above didnt work so get 2 months before
                            if (lag3 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)


                        #Test 2
                        if(img_day_int >= 15 and img_month != '12'):
                            #just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                           # above didnt work so get the month before
                            if (lag3 == None):
                                # check if the month is January
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            # above didnt work so get 2 months after
                            if (lag3 == None):
                                if (img_month == '11'):
                                    # add a year and make January
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br',
                                                     band_index=band_index, ls=7)

                                # add two months
                                month_plus_2 = str(img_month_int + 1)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2

                                updated_name = pattern1 + str(year_minus_3) + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            # above didnt work so get 2 months before
                            if (lag3 == None):
                                # check to see if the month is Jan or Feb
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to november
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '11' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                if (img_month == '02'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)



                        #Test 3
                        if(img_day_int < 15 and img_month == '01'):
                            #subtract a year
                            #change month to december
                            year_minus_3_minus_1 = str(year_minus_3 - 1)

                            updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                            # above didnt work so add a month
                            if (lag3 == None):
                                # just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            # above didnt work so subtract 2 months
                            if (lag3 == None):
                                # subtract a year
                                # change month to november
                                year_minus_3_minus_1 = str(year_minus_3 - 1)

                                updated_name = pattern1 + year_minus_3_minus_1 + '11' + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            # above didnt work so add 2 months
                            if (lag3 == None):
                                # b/c its jan just add 2 months
                                month_plus_1 = str(img_month_int + 2)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)


                        #Test 4
                        if(img_day_int < 15 and img_month != '01'):
                            #subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_3_str + updated_month + pattern2


                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                            #above didnt work so get one month after
                            if(lag3 == None):
                                #check for december
                                if(img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                #Just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                                #rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                            # above didnt work so get 2 months before
                            if (lag3 == None):
                                #check for feburary
                                if(img_month == '02'):
                                    # change month to december
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                     band_index=band_index, ls=7)

                                #subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)

                            #above didnt work so add 2 months
                            if(lag3 == None):
                                # check for november
                                if (img_month == '11'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                #check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=7)

                                # Just add 2 months
                                month_plus_2 = str(img_month_int + 2)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2
                                updated_name = pattern1 + yr_min_3_str + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=7)


                            #rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br', band_index=band_index, ls=7)

                    #make the dataset for the lags from the correct file paths
                    l1_ds = ind.read_data(lag1, raster_driver_name='GTiff')
                    l2_ds = ind.read_data(lag2, raster_driver_name='GTiff')
                    l3_ds = ind.read_data(lag3, raster_driver_name='GTiff')

                    #make the bands
                    l1_band = l1_ds.GetRasterBand(band_index)
                    l2_band = l2_ds.GetRasterBand(band_index)
                    l3_band = l3_ds.GetRasterBand(band_index)

                    #compute the operation
                    if(operation == 'avg'):
                        #Generate the avg lag data
                        operation_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                    elif(operation == 'std'):
                        # this is sqrt(sigma(value - mean) / number of instances )
                        #I am using sample standard deviation
                        #calculate the average
                        avgerage_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                        #calculate the StDev
                        operation_img = np.sqrt(
                            (
                                (l1_band.ReadAsArray() - avgerage_img) +
                                (l2_band.ReadAsArray() - avgerage_img) +
                                (l3_band.ReadAsArray() - avgerage_img)
                            ) / 2
                        )

                    elif(operation == 'dif'):
                        #get the lag average
                        avgerage_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                        #Call the origional image and make into a raster np array that can be manipulated
                        og_img = dataset.GetRasterBand(band_index)

                        #Calculate the differenced SVI (pre - post fire aka avg - og)
                        operation_img = avgerage_img - og_img.ReadAsArray()

                    else:
                        print("Pick either 'std', 'avg', or 'dif'")

                    #I am going to assume all the files have the same numbers of columns and rows

                    #Make the output filepath
                    # out_name = band_name + pattern1 + img_year + img_month + pattern2
                    # out_name = band_name + '_' + band_name
                    out_name = '{}_{}_{}.tif'.format(out_sat, band_name, out_date)

                    # Use OS path join
                    out_fp = os.path.join(out_path, out_name)
                    # out_fp = out_path + out_name



                    # Save the image to file
                    ind.output_single_band_raster(operation_img, out_fp, cols, rows, num_band=1,
                                                  raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                                                  metadata=None, nodataval=-9999)



                # I want to see how long the process takes
                time_dif = time.time() - start_time
                iteration_num += 1
                print('Iteration {} took {} seconds to complete'.format(iteration_num, time_dif))

    # except Exception as e:
    #     print(e)


#=======================================================================================================================
#                   Landsat 5
#=======================================================================================================================

def l5_lag_svi_avg(in_path, out_path, band_index, band_name, MIN_YEAR_5, LAG_YEAR, operation='avg'):
    """
    Makes the lag average and dSVI for the images
    :param in_path: in data path, string, ex: './data/'
    :param out_path: out data path, string, ex: './output/'
    :param band_index: the index of the band of interest, integer
    :param band_name: the name of the SVI of interest, string
    :param operation: the name of the desired operation. Options: avg, std, or dif, string
    :return: Average, STDEV, and/or dSVI for specific SVIs
    """

    #helps with errors
    # try:
    #something silly to nullify the try statement/ keep the index
    if(1 == 1):

        # I want to see how long the process takes, this will mark the iterations
        iteration_num = 0

        files = os.listdir(in_path)

        #read through the files in the directory
        for file in os.listdir(in_path):
            if file.endswith('.tif'):

                #make the full path
                in_image = in_path + file

                #I want to see how long the process takes
                start_time = time.time()

                #make the dataset by reading in the data
                dataset = ind.read_data(in_image, raster_driver_name='GTiff')

                #get the projection
                projection = dataset.GetProjection()

                #get the geotransform
                geotransform = dataset.GetGeoTransform()

                #Get indexed raster band
                band = dataset.GetRasterBand(band_index)

                #get the no data value
                no_data_value = band.GetNoDataValue()

                #get the dimensions of the image
                cols = dataset.RasterXSize
                rows = dataset.RasterYSize


                #If statement for Landsat 8
                #checks if the top left and bottom left corners have data
                c1 = corner_value(dataset, band_index, rows, cols, 'tl')
                c2 = corner_value(dataset, band_index, rows, cols, 'bl')
                if (np.isnan(c1[0,0]) == False and np.isnan(c2[0,0]) == False):

                    #get the landsat name
                    out_sat = file[0:4]

                    #get the full input data for the output name
                    out_date = file[15:23]

                    #Get the date of the image
                    img_year = file[15:19]
                    #make it an integer
                    img_year_int = int(img_year)

                    #This line skips years that don't meet the criteria
                    if img_year_int < MIN_YEAR_5 + int(LAG_YEAR):
                        continue


                    #get the month
                    img_month = file[19:21]
                    img_month_int =int(img_month)
                    #get the days
                    img_day = file[21:23]
                    img_day_int = int(img_day)

                    #get the three year lag of the selected image
                    year_minus_1 = img_year_int - 1
                    year_minus_2 = img_year_int - 2
                    year_minus_3 = img_year_int - 3
                    #convert to string
                    yr_min_1_str = str(year_minus_1)
                    yr_min_2_str = str(year_minus_2)
                    yr_min_3_str = str(year_minus_3)

                    # LC08_CU_004013_20130324.tif
                    # expression pattern for landsat8
                    pattern1 = 'L..5.CU.004013.'
                    pattern2 = '...tif' #date and the .tif ending
                    pattern_full_yr1 = pattern1 + yr_min_1_str + img_month + pattern2
                    pattern_full_yr2 = pattern1 + yr_min_2_str + img_month + pattern2
                    pattern_full_yr3 = pattern1 + yr_min_3_str + img_month + pattern2
                    #* Above should work with regular expressions but it may need to be modified
                    # *** double check to see if the regular expressions work


                    #Test to see if the lag data exists for the month, if not get it from an adjacent month

                    #Lag 1
                    lag1 = find_file(file_name=pattern_full_yr1, file_path=in_path, corner1='tl', corner2='bl',
                                     band_index=band_index)
                    lag1_test = lag1
                    if lag1_test == None:
                        # Test 1
                        # check the day of the first image
                        if (img_day_int >= 15 and img_month == '12'):
                            # add a year and make january
                            yr_plus_1 = str(year_minus_1 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            # rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=5)

                            # above didnt work so get the month before
                            if (lag1 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months after
                            if (lag1 == None):
                                # add a year and make Feburary
                                yr_plus_1 = str(year_minus_1 + 1)
                                updated_name = pattern1 + str(yr_plus_1) + '02' + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag1 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                        # Test 2
                        if (img_day_int >= 15 and img_month != '12'):
                            # just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                            # rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=5)

                            # above didnt work so get the month before
                            if (lag1 == None):
                                # check if the month is January
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months after
                            if (lag1 == None):
                                if (img_month == '11'):
                                    # add a year and make January
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl',
                                                     band_index=band_index, ls=5)

                                # add two months
                                month_plus_2 = str(img_month_int + 1)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2

                                updated_name = pattern1 + str(year_minus_1) + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag1 == None):
                                # check to see if the month is Jan or Feb
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to november
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '11' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                if (img_month == '02'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                    # subtract 2 months
                                    updated_month = str(img_month_int - 2)
                                    if int(updated_month) < 10:
                                        updated_month = '0' + updated_month
                                    updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl',
                                                     band_index=band_index, ls=5)

                        # Test 3
                        if (img_day_int < 15 and img_month == '01'):
                            # subtract a year
                            # change month to december
                            year_minus_1_minus_1 = str(year_minus_1 - 1)

                            updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                            # rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=5)

                            # above didnt work so add a month
                            if (lag1 == None):
                                # just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so subtract 2 months
                            if (lag1 == None):
                                # subtract a year
                                # change month to november
                                year_minus_1_minus_1 = str(year_minus_1 - 1)

                                updated_name = pattern1 + year_minus_1_minus_1 + '11' + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so add 2 months
                            if (lag1 == None):
                                # b/c its jan just add 2 months
                                month_plus_1 = str(img_month_int + 2)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                        # Test 4
                        if (img_day_int < 15 and img_month != '01'):
                            # subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                            # rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=5)

                            # above didnt work so get one month after
                            if (lag1 == None):
                                # check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # Just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag1 == None):
                                # check for feburary
                                if (img_month == '02'):
                                    # change month to december
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl',
                                                     band_index=band_index, ls=5)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so add 2 months
                            if (lag1 == None):
                                # check for november
                                if (img_month == '11'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # Just add 2 months
                                month_plus_2 = str(img_month_int + 2)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2
                                updated_name = pattern1 + yr_min_1_str + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                    # Lag 2
                    lag2 = find_file(file_name=pattern_full_yr2, file_path=in_path, corner1='tl', corner2='bl',
                                     band_index=band_index, ls=5)
                    lag2_test = lag2
                    if lag2_test == None:
                        # Test 1
                        # check the day of the first image
                        if (img_day_int >= 15 and img_month == '12'):
                            # add a year and make january
                            yr_plus_1 = str(year_minus_2 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            # rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=5)

                            # above didnt work so get the month before
                            if (lag2 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months after
                            if (lag2 == None):
                                # add a year and make Feburary
                                yr_plus_1 = str(year_minus_2 + 1)
                                updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag2 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                        # Test 2
                        if (img_day_int >= 15 and img_month != '12'):
                            # just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                            # rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=5)

                            # above didnt work so get the month before
                            if (lag2 == None):
                                # check if the month is January
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months after
                            if (lag2 == None):
                                if (img_month == '11'):
                                    # add a year and make January
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl',
                                                     band_index=band_index, ls=5)

                                # add two months
                                month_plus_2 = str(img_month_int + 1)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2

                                updated_name = pattern1 + str(year_minus_2) + str(month_plus_2) + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag2 == None):
                                # check to see if the month is Jan or Feb
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to november
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '11' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                if (img_month == '02'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                        # Test 3
                        if (img_day_int < 15 and img_month == '01'):
                            # subtract a year
                            # change month to december
                            year_minus_2_minus_1 = str(year_minus_2 - 1)

                            updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                            # rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=5)

                            # above didnt work so add a month
                            if (lag2 == None):
                                # just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so subtract 2 months
                            if (lag2 == None):
                                # subtract a year
                                # change month to november
                                year_minus_2_minus_1 = str(year_minus_2 - 1)

                                updated_name = pattern1 + year_minus_2_minus_1 + '11' + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so add 2 months
                            if (lag2 == None):
                                # b/c its jan just add 2 months
                                month_plus_1 = str(img_month_int + 2)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                        # Test 4
                        if (img_day_int < 15 and img_month != '01'):
                            # subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                            # rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=5)

                            # above didnt work so get one month after
                            if (lag2 == None):
                                # check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # Just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag2 == None):
                                # check for feburary
                                if (img_month == '02'):
                                    # change month to december
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl',
                                                     band_index=band_index, ls=5)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so add 2 months
                            if (lag2 == None):
                                # check for november
                                if (img_month == '11'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # Just add 2 months
                                month_plus_2 = str(img_month_int + 2)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2
                                updated_name = pattern1 + yr_min_2_str + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                    # Lag 3
                    lag3 = find_file(file_name=pattern_full_yr3, file_path=in_path, corner1='tl', corner2='bl',
                                     band_index=band_index, ls=5)
                    lag3_test = lag3
                    if lag3_test == None:
                        # Test1
                        # check the day of the first image
                        if (img_day_int >= 15 and img_month == '12'):
                            # add a year and make january
                            yr_plus_1 = str(year_minus_3 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            # rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=5)

                            # above didnt work so get the month before
                            if (lag3 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                la3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                band_index=band_index, ls=5)

                            # above didnt work so get 2 months after
                            if (lag3 == None):
                                # add a year and make Feburary
                                yr_plus_1 = str(year_minus_3 + 1)
                                updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag3 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                        # Test 2
                        if (img_day_int >= 15 and img_month != '12'):
                            # just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                            # rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=5)

                            # above didnt work so get the month before
                            if (lag3 == None):
                                # check if the month is January
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + str(yr_min_3_str) + str(updated_month) + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months after
                            if (lag3 == None):
                                if (img_month == '11'):
                                    # add a year and make January
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl',
                                                     band_index=band_index, ls=5)

                                # add two months
                                month_plus_2 = str(img_month_int + 1)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2

                                updated_name = pattern1 + str(year_minus_3) + str(month_plus_2) + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag3 == None):
                                # check to see if the month is Jan or Feb
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to november
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '11' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                if (img_month == '02'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + str(yr_min_3_str) + str(updated_month) + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                        # Test 3
                        if (img_day_int < 15 and img_month == '01'):
                            # subtract a year
                            # change month to december
                            year_minus_3_minus_1 = str(year_minus_3 - 1)

                            updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                            # rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=5)

                            # above didnt work so add a month
                            if (lag3 == None):
                                # just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + str(yr_min_3_str) + str(month_plus_1) + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so subtract 2 months
                            if (lag3 == None):
                                # subtract a year
                                # change month to november
                                year_minus_3_minus_1 = str(year_minus_3 - 1)

                                updated_name = pattern1 + year_minus_3_minus_1 + '11' + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so add 2 months
                            if (lag3 == None):
                                # b/c its jan just add 2 months
                                month_plus_1 = str(img_month_int + 2)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                        # Test 4
                        if (img_day_int < 15 and img_month != '01'):
                            # subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                            # rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                             band_index=band_index, ls=5)

                            # above didnt work so get one month after
                            if (lag3 == None):
                                # check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # Just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag3 == None):
                                # check for feburary
                                if (img_month == '02'):
                                    # change month to december
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl',
                                                     band_index=band_index, ls=5)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                            # above didnt work so add 2 months
                            if (lag3 == None):
                                # check for november
                                if (img_month == '11'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl',
                                                     corner2='bl', band_index=band_index, ls=5)

                                # Just add 2 months
                                month_plus_2 = str(img_month_int + 2)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2
                                updated_name = pattern1 + yr_min_3_str + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tl', corner2='bl',
                                                 band_index=band_index, ls=5)

                    #make the dataset for the lags from the correct file paths
                    l1_ds = ind.read_data(lag1, raster_driver_name='GTiff')
                    l2_ds = ind.read_data(lag2, raster_driver_name='GTiff')
                    l3_ds = ind.read_data(lag3, raster_driver_name='GTiff')

                    #make the bands
                    l1_band = l1_ds.GetRasterBand(band_index)
                    l2_band = l2_ds.GetRasterBand(band_index)
                    l3_band = l3_ds.GetRasterBand(band_index)

                    #compute the operation
                    if(operation == 'avg'):
                        #Generate the avg lag data
                        operation_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                    elif(operation == 'std'):
                        # this is sqrt(sigma(value - mean) / number of instances )
                        #I am using sample standard deviation
                        #calculate the average
                        avgerage_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                        #calculate the StDev
                        operation_img = np.sqrt(
                            (
                                (l1_band.ReadAsArray() - avgerage_img) +
                                (l2_band.ReadAsArray() - avgerage_img) +
                                (l3_band.ReadAsArray() - avgerage_img)
                            ) / 2
                        )

                    elif(operation == 'dif'):
                        #get the lag average
                        avgerage_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                        #Call the origional image and make into a raster np array that can be manipulated
                        og_img = dataset.GetRasterBand(band_index)

                        #Calculate the differenced SVI (pre - post fire aka avg - og)
                        operation_img = avgerage_img - og_img.ReadAsArray()

                    else:
                        print("Pick either 'std', 'avg', or 'dif'")

                    #I am going to assume all the files have the same numbers of columns and rows

                    #Make the output filepath
                    # out_name = band_name + pattern1 + img_year + img_month + pattern2
                    # out_name = band_name + '_' + band_name
                    out_name = '{}_{}_{}.tif'.format(out_sat, band_name, out_date)

                    #Use OS path join
                    out_fp = os.path.join(out_path, out_name)
                    # out_fp = out_path + out_name



                    #Save the image to file
                    ind.output_single_band_raster(operation_img, out_fp, cols, rows, num_band=1,
                                                  raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                                                  metadata=None, nodataval=-9999)




                # checks if the top right and bottom right corners have data
                c1 = corner_value(dataset, band_index, rows, cols, 'tr')
                c2 = corner_value(dataset, band_index, rows, cols, 'br')
                if (np.isnan(c1[0,0]) == False and np.isnan(c2[0,0]) == False):

                    #get the landsat name
                    out_sat = file[0:4]

                    #get the full input data for the output name
                    out_date = file[15:23]

                    #Get the date of the image
                    img_year = file[15:19]
                    #make it an integer
                    img_year_int = int(img_year)

                    #This line skips years that don't meet the criteria
                    if img_year_int < MIN_YEAR_5 + int(LAG_YEAR):
                        continue


                    #get the month
                    img_month = file[19:21]
                    img_month_int =int(img_month)
                    #get the days
                    img_day = file[21:23]
                    img_day_int = int(img_day)

                    #get the three year lag of the selected image
                    year_minus_1 = img_year_int - 1
                    year_minus_2 = img_year_int - 2
                    year_minus_3 = img_year_int - 3
                    #convert to string
                    yr_min_1_str = str(year_minus_1)
                    yr_min_2_str = str(year_minus_2)
                    yr_min_3_str = str(year_minus_3)

                    # LC08_CU_004013_20130324.tif
                    # expression pattern for landsat8
                    pattern1 = 'L..5.CU.004013.'
                    pattern2 = '...tif' #date and the .tif ending
                    pattern_full_yr1 = pattern1 + yr_min_1_str + img_month + pattern2
                    pattern_full_yr2 = pattern1 + yr_min_2_str + img_month + pattern2
                    pattern_full_yr3 = pattern1 + yr_min_3_str + img_month + pattern2
                    #* Above should work with regular expressions but it may need to be modified
                    # *** double check to see if the regular expressions work


                    #Test to see if the lag data exists for the month, if not get it from an adjacent month

                    #Lag 1
                    lag1 = find_file(file_name=pattern_full_yr1, file_path=in_path, corner1='tr', corner2='br',
                                     band_index=band_index, ls=5)
                    lag1_test = lag1
                    if lag1_test == None:
                        # Test 1
                        # check the day of the first image
                        if (img_day_int >= 15 and img_month == '12'):
                            # add a year and make january
                            yr_plus_1 = str(year_minus_1 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            # rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                            # above didnt work so get the month before
                            if (lag1 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months after
                            if (lag1 == None):
                                # add a year and make Feburary
                                yr_plus_1 = str(year_minus_1 + 1)
                                updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag1 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                        # Test 2
                        if (img_day_int >= 15 and img_month != '12'):
                            # just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                            # rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                            # above didnt work so get the month before
                            if (lag1 == None):
                                # check if the month is January
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months after
                            if (lag1 == None):
                                if (img_month == '11'):
                                    # add a year and make January
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br',
                                                     band_index=band_index, ls=5)

                                # add two months
                                month_plus_2 = str(img_month_int + 1)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2

                                updated_name = pattern1 + str(year_minus_1) + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag1 == None):
                                # check to see if the month is Jan or Feb
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to november
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '11' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                if (img_month == '02'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                    # subtract 2 months
                                    updated_month = str(img_month_int - 2)
                                    if int(updated_month) < 10:
                                        updated_month = '0' + updated_month
                                    updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br',
                                                     band_index=band_index, ls=5)

                        # Test 3
                        if (img_day_int < 15 and img_month == '01'):
                            # subtract a year
                            # change month to december
                            year_minus_1_minus_1 = str(year_minus_1 - 1)

                            updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                            # rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                            # above didnt work so add a month
                            if (lag1 == None):
                                # just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so subtract 2 months
                            if (lag1 == None):
                                # subtract a year
                                # change month to november
                                year_minus_1_minus_1 = str(year_minus_1 - 1)

                                updated_name = pattern1 + year_minus_1_minus_1 + '11' + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so add 2 months
                            if (lag1 == None):
                                # b/c its jan just add 2 months
                                month_plus_1 = str(img_month_int + 2)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                        # Test 4
                        if (img_day_int < 15 and img_month != '01'):
                            # subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                            # rename the updated find file data
                            lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                            # above didnt work so get one month after
                            if (lag1 == None):
                                # check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # Just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_1_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag1 == None):
                                # check for feburary
                                if (img_month == '02'):
                                    # change month to december
                                    year_minus_1_minus_1 = str(year_minus_1 - 1)

                                    updated_name = pattern1 + year_minus_1_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br',
                                                     band_index=band_index, ls=5)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_1_str + updated_month + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so add 2 months
                            if (lag1 == None):
                                # check for november
                                if (img_month == '11'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_1 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                    # rename the updated find file data
                                    lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # Just add 2 months
                                month_plus_2 = str(img_month_int + 2)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2
                                updated_name = pattern1 + yr_min_1_str + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag1 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                    # Lag 2
                    lag2 = find_file(file_name=pattern_full_yr2, file_path=in_path, corner1='tr', corner2='br',
                                     band_index=band_index, ls=5)
                    lag2_test = lag2
                    if lag2_test == None:
                        # Test 1
                        # check the day of the first image
                        if (img_day_int >= 15 and img_month == '12'):
                            # add a year and make january
                            yr_plus_1 = str(year_minus_2 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            # rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                            # above didnt work so get the month before
                            if (lag2 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months after
                            if (lag2 == None):
                                # add a year and make Feburary
                                yr_plus_1 = str(year_minus_2 + 1)
                                updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag2 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                        # Test 2
                        if (img_day_int >= 15 and img_month != '12'):
                            # just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                            # rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                            # above didnt work so get the month before
                            if (lag2 == None):
                                # check if the month is January
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months after
                            if (lag2 == None):
                                if (img_month == '11'):
                                    # add a year and make January
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br',
                                                     band_index=band_index, ls=5)

                                # add two months
                                month_plus_2 = str(img_month_int + 1)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2

                                updated_name = pattern1 + str(year_minus_2) + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag2 == None):
                                # check to see if the month is Jan or Feb
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to november
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + str(year_minus_2_minus_1) + '11' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                if (img_month == '02'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                        # Test 3
                        if (img_day_int < 15 and img_month == '01'):
                            # subtract a year
                            # change month to december
                            year_minus_2_minus_1 = str(year_minus_2 - 1)

                            updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                            # rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                            # above didnt work so add a month
                            if (lag2 == None):
                                # just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so subtract 2 months
                            if (lag2 == None):
                                # subtract a year
                                # change month to november
                                year_minus_2_minus_1 = str(year_minus_2 - 1)

                                updated_name = pattern1 + year_minus_2_minus_1 + '11' + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so add 2 months
                            if (lag2 == None):
                                # b/c its jan just add 2 months
                                month_plus_1 = str(img_month_int + 2)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                        # Test 4
                        if (img_day_int < 15 and img_month != '01'):
                            # subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                            # rename the updated find file data
                            lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                            # above didnt work so get one month after
                            if (lag2 == None):
                                # check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # Just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_2_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag2 == None):
                                # check for feburary
                                if (img_month == '02'):
                                    # change month to december
                                    year_minus_2_minus_1 = str(year_minus_2 - 1)

                                    updated_name = pattern1 + year_minus_2_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br',
                                                     band_index=band_index, ls=5)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_2_str + updated_month + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so add 2 months
                            if (lag2 == None):
                                # check for november
                                if (img_month == '11'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_2 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                    # rename the updated find file data
                                    lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # Just add 2 months
                                month_plus_2 = str(img_month_int + 2)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2
                                updated_name = pattern1 + yr_min_2_str + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag2 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                    # Lag 3
                    lag3 = find_file(file_name=pattern_full_yr3, file_path=in_path, corner1='tr', corner2='br',
                                     band_index=band_index, ls=5)
                    lag3_test = lag3
                    if lag3_test == None:
                        # Test1
                        # check the day of the first image
                        if (img_day_int >= 15 and img_month == '12'):
                            # add a year and make january
                            yr_plus_1 = str(year_minus_3 + 1)
                            updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                            # rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                            # above didnt work so get the month before
                            if (lag3 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                la3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                band_index=band_index, ls=5)

                            # above didnt work so get 2 months after
                            if (lag3 == None):
                                # add a year and make Feburary
                                yr_plus_1 = str(year_minus_3 + 1)
                                updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag3 == None):
                                # subtract a month
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                        # Test 2
                        if (img_day_int >= 15 and img_month != '12'):
                            # just add a month
                            month_plus_1 = str(img_month_int + 1)
                            if int(month_plus_1) < 10:
                                month_plus_1 = '0' + month_plus_1
                            updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                            # rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                            # above didnt work so get the month before
                            if (lag3 == None):
                                # check if the month is January
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # subtract a month
                                updated_month = str(img_month_int - 1)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months after
                            if (lag3 == None):
                                if (img_month == '11'):
                                    # add a year and make January
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br',
                                                     band_index=band_index, ls=5)

                                # add two months
                                month_plus_2 = str(img_month_int + 1)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2

                                updated_name = pattern1 + str(year_minus_3) + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag3 == None):
                                # check to see if the month is Jan or Feb
                                if (img_month == '01'):
                                    # subtract a year
                                    # change month to november
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '11' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                if (img_month == '02'):
                                    # subtract a year
                                    # change month to december
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                        # Test 3
                        if (img_day_int < 15 and img_month == '01'):
                            # subtract a year
                            # change month to december
                            year_minus_3_minus_1 = str(year_minus_3 - 1)

                            updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                            # rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                            # above didnt work so add a month
                            if (lag3 == None):
                                # just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so subtract 2 months
                            if (lag3 == None):
                                # subtract a year
                                # change month to november
                                year_minus_3_minus_1 = str(year_minus_3 - 1)

                                updated_name = pattern1 + year_minus_3_minus_1 + '11' + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so add 2 months
                            if (lag3 == None):
                                # b/c its jan just add 2 months
                                month_plus_1 = str(img_month_int + 2)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                        # Test 4
                        if (img_day_int < 15 and img_month != '01'):
                            # subtract a month

                            updated_month = str(img_month_int - 1)
                            if int(updated_month) < 10:
                                updated_month = '0' + updated_month
                            updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                            # rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                            # above didnt work so get one month after
                            if (lag3 == None):
                                # check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # Just add a month
                                month_plus_1 = str(img_month_int + 1)
                                if int(month_plus_1) < 10:
                                    month_plus_1 = '0' + month_plus_1
                                updated_name = pattern1 + yr_min_3_str + month_plus_1 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so get 2 months before
                            if (lag3 == None):
                                # check for feburary
                                if (img_month == '02'):
                                    # change month to december
                                    year_minus_3_minus_1 = str(year_minus_3 - 1)

                                    updated_name = pattern1 + year_minus_3_minus_1 + '12' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br',
                                                     band_index=band_index, ls=5)

                                # subtract 2 months
                                updated_month = str(img_month_int - 2)
                                if int(updated_month) < 10:
                                    updated_month = '0' + updated_month
                                updated_name = pattern1 + yr_min_3_str + updated_month + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # above didnt work so add 2 months
                            if (lag3 == None):
                                # check for november
                                if (img_month == '11'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '01' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # check for december
                                if (img_month == '12'):
                                    # add a year and make january
                                    yr_plus_1 = str(year_minus_3 + 1)
                                    updated_name = pattern1 + yr_plus_1 + '02' + pattern2

                                    # rename the updated find file data
                                    lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr',
                                                     corner2='br', band_index=band_index, ls=5)

                                # Just add 2 months
                                month_plus_2 = str(img_month_int + 2)
                                if int(month_plus_2) < 10:
                                    month_plus_2 = '0' + month_plus_2
                                updated_name = pattern1 + yr_min_3_str + month_plus_2 + pattern2

                                # rename the updated find file data
                                lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                                 band_index=band_index, ls=5)

                            # rename the updated find file data
                            lag3 = find_file(file_name=updated_name, file_path=in_path, corner1='tr', corner2='br',
                                             band_index=band_index, ls=5)

                    #make the dataset for the lags from the correct file paths
                    l1_ds = ind.read_data(lag1, raster_driver_name='GTiff')
                    l2_ds = ind.read_data(lag2, raster_driver_name='GTiff')
                    l3_ds = ind.read_data(lag3, raster_driver_name='GTiff')

                    #make the bands
                    l1_band = l1_ds.GetRasterBand(band_index)
                    l2_band = l2_ds.GetRasterBand(band_index)
                    l3_band = l3_ds.GetRasterBand(band_index)

                    #compute the operation
                    if(operation == 'avg'):
                        #Generate the avg lag data
                        operation_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                    elif(operation == 'std'):
                        # this is sqrt(sigma(value - mean) / number of instances )
                        #I am using sample standard deviation
                        #calculate the average
                        avgerage_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                        #calculate the StDev
                        operation_img = np.sqrt(
                            (
                                (l1_band.ReadAsArray() - avgerage_img) +
                                (l2_band.ReadAsArray() - avgerage_img) +
                                (l3_band.ReadAsArray() - avgerage_img)
                            ) / 2
                        )

                    elif(operation == 'dif'):
                        #get the lag average
                        avgerage_img = (l1_band.ReadAsArray() + l2_band.ReadAsArray() + l3_band.ReadAsArray()) / 3

                        #Call the origional image and make into a raster np array that can be manipulated
                        og_img = dataset.GetRasterBand(band_index)

                        #Calculate the differenced SVI (pre - post fire aka avg - og)
                        operation_img = avgerage_img - og_img.ReadAsArray()

                    else:
                        print("Pick either 'std', 'avg', or 'dif'")


                    #I am going to assume all the files have the same numbers of columns and rows

                    #Make the output filepath
                    # out_name = band_name + pattern1 + img_year + img_month + pattern2
                    # out_name = band_name + '_' + band_name
                    out_name = '{}_{}_{}.tif'.format(out_sat, band_name, out_date)

                    # Use OS path join
                    out_fp = os.path.join(out_path, out_name)
                    # out_fp = out_path + out_name



                    # Save the image to file
                    ind.output_single_band_raster(operation_img, out_fp, cols, rows, num_band=1,
                                                  raster_driver_name='GTiff', projection=projection, geotransform=geotransform,
                                                  metadata=None, nodataval=-9999)



                # I want to see how long the process takes
                time_dif = time.time() - start_time
                iteration_num += 1
                print('Iteration {} took {} seconds to complete'.format(iteration_num, time_dif))

    #except Exception as e:
     #   print(e)


#=======================================================================================================================
#                   List of indices that need to be run
#=======================================================================================================================

"""
Lag SVIs (6) that I will need for the input data with the input index
dNBR2       [7]     dif_nbr2
dVI57       [17]    dif_vi57
NDMI_AVG    [9]     avg_ndmi
GEMI_AVG    [4]     avg_gemi
VI46_AVG    [16]    avg_vi46
VI6T_STD    [13]    std_vi6t
"""

#=======================================================================================================================
#                   Parallization
#=======================================================================================================================

def combined_function(d):

    if(d['satellite'] == 8):
        l8_lag_svi_avg(d['in_path'], d['out_path'], d['band_index'], d['band_name'], d['MIN_YEAR_8'], d['LAG_YEAR'], d['operation'])

    elif(d['satellite'] == 7):
        l7_lag_svi_avg(d['in_path'], d['out_path'], d['band_index'], d['band_name'], d['MIN_YEAR_7'], d['LAG_YEAR'], d['operation'])

    elif(d['satellite'] == 5):
        l5_lag_svi_avg(d['in_path'], d['out_path'], d['band_index'], d['band_name'], d['MIN_YEAR_5'], d['LAG_YEAR'], d['operation'])

    else:
        print('pick another option: 8 for ls 8, 7 for ls 7, 5 for ls 5')


#=======================================================================================================================
#                   Code from the lag maker parallel
#=======================================================================================================================


#This is what makes it parallel
def parallel_raster(ary_d):
    num_process = len(ary_d)
    print("# of CPUs = {}".format(cpu_count()))
    print("# of processes = {}".format(num_process))

    processes = []
    start = time.time()

    for i in range(num_process):
        p = Process(target=combined_function, args=(ary_d[i],))

        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    end = time.time() - start
    minutes = end/60
    hours = minutes/60
    print("parallel completed in {} seconds AKA {} minutes AKA {} hours".format(end, minutes, hours))



#=======================================================================================================================
#                   Run the script
#=======================================================================================================================


if __name__ == '__main__':
    #Time for the entire process
    allTimeStart = time.time()

    #make sure to change this depending on what satellite I am using
    MIN_YEAR_5 = 2008
    MIN_YEAR_7 = 2000
    MIN_YEAR_8 = 2014
    LAG_YEAR = 3
    #For LS 8
    # l8_lag_svi_avg(in_path='F:/Thesis/svi_stack/Landsat_8/',
    #               out_path='//geo-procella/GradData/cross8046/l8_lag_single_band_indices/',
    #               band_index=13,
    #               band_name='std_vi6t',
    #               operation='std')

    # l8_lag_svi_avg(in_path='D:/svi_stack/Landsat_8/',
    #               out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/l8/',
    #               band_index=7,
    #               band_name='dif_nbr2',
    #               operation='dif')

    # l8_lag_svi_avg(in_path='D:/svi_stack/Landsat_8/',
    #               out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/l8/',
    #               band_index=17,
    #               band_name='dif_vi57',
    #               operation='dif')

    #For ls 7
    '''
    l7_lag_svi_avg(#in_path='F:/Thesis/svi_stack/Landsat_7/',
                  # in_path='C:/Users/eduroscha001/Documents/thesis/svi_stack/',
                  in_path='D:/svi_stack/Landsat_7/',
                  # out_path='//geo-procella/GradData/cross8046/l7_lag_single_band_indices/',
                  # out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/',
                  out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/l7/fix/',
                  band_index=7,
                  band_name='dif_nbr2',
                  operation='dif')'''

    # l7_lag_svi_avg(in_path='D:/svi_stack/Landsat_7/',
    #                out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/l7/',
    #                band_index=17,
    #                band_name='dif_vi57',
    #                operation='dif')


    # l7_lag_svi_avg(in_path='C:/Users/eduroscha001/Documents/thesis/svi_stack/',
    #                out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/',
    #                band_index=9,
    #                band_name='avg_ndmi',
    #                operation='avg')
    #
    # l7_lag_svi_avg(in_path='C:/Users/eduroscha001/Documents/thesis/svi_stack/',
    #                out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/',
    #                band_index=4,
    #                band_name='avg_gemi',
    #                operation='avg')
    #
    # l7_lag_svi_avg(in_path='C:/Users/eduroscha001/Documents/thesis/svi_stack/',
    #                out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/',
    #                band_index=16,
    #                band_name='avg_vi46',
    #                operation='avg')
    #
    # l7_lag_svi_avg(in_path='C:/Users/eduroscha001/Documents/thesis/svi_stack/',
    #                out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/',
    #                band_index=13,
    #                band_name='std_vi6t',
    #                operation='std')



    # for ls 5
    # l5_lag_svi_avg(in_path='D:/svi_stack/Landsat_5/',
    #               out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/l5/',
    #               band_index=7,
    #               band_name='dif_nbr2',
    #               operation='dif')

    # l5_lag_svi_avg(in_path='D:/svi_stack/Landsat_5/',
    #               out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/l5/',
    #               band_index=17,
    #               band_name='dif_vi57',
    #               operation='dif')

#    l5_lag_svi_avg(in_path='C:/Users/eduroscha001/Documents/thesis/svi_stack/',
#                   out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/',
#                   band_index=9,
#                   band_name='avg_ndmi',
#                   operation='avg')

#    l5_lag_svi_avg(in_path='C:/Users/eduroscha001/Documents/thesis/svi_stack/',
#                   out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/',
#                   band_index=4,
#                   band_name='avg_gemi',
#                   operation='avg')

#    l5_lag_svi_avg(in_path='C:/Users/eduroscha001/Documents/thesis/svi_stack/',
#                   out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/',
#                   band_index=16,
#                   band_name='avg_vi46',
#                   operation='avg')

#    l5_lag_svi_avg(in_path='C:/Users/eduroscha001/Documents/thesis/svi_stack/',
#                   out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/',
#                   band_index=13,
#                   band_name='std_vi6t',
#                   operation='std')

    # For ls 7

    # l7_lag_svi_avg(#in_path='F:/Thesis/svi_stack/Landsat_7/',
                  # in_path='C:/Users/eduroscha001/Documents/thesis/svi_stack/',
                  # in_path='D:/svi_stack/Landsat_7/',
                  # out_path='//geo-procella/GradData/cross8046/l7_lag_single_band_indices/',
                  # out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/',
                  # out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/l7/fix/',
                  # band_index=7,
                  # band_name='dif_nbr2',
                  # operation='dif')

    # l7_lag_svi_avg(in_path='D:/svi_stack/Landsat_7/',
    #                out_path='C:/Users/eduroscha001/Documents/thesis/lag_data/l7/',
    #                band_index=17,
    #                band_name='dif_vi57',
    #                operation='dif')


    l8_1 = {'in_path': 'F:/Thesis/svi_stack/Landsat_8/',
          'out_path': 'C:/Users/eduroscha001/Documents/thesis/lag_data/l8/',
          'band_index': 7,
          'band_name':'dif_nbr2',
          'MIN_YEAR_8': 2014,
          'LAG_YEAR': 3,
          'operation':'dif',
          'satellite': 8}

    l8_2 = {'in_path': 'F:/Thesis/svi_stack/Landsat_8/',
          'out_path': 'C:/Users/eduroscha001/Documents/thesis/lag_data/l8/',
          'band_index': 17,
          'band_name':'dif_vi57',
          'operation':'dif',
          'MIN_YEAR_8': 2014,
          'LAG_YEAR': 3,
          'satellite': 8}

    l7_1 = {'in_path': 'D:/svi_stack/Landsat_7/',
          'out_path': 'C:/Users/eduroscha001/Documents/thesis/lag_data/l7/',
          'band_index': 7,
          'band_name': 'dif_nbr2',
          'MIN_YEAR_7': 2013,
          'LAG_YEAR': 3,
          'operation': 'dif',
          'satellite': 7}

    l7_2 = {'in_path': 'D:/svi_stack/Landsat_7/',
          'out_path': 'C:/Users/eduroscha001/Documents/thesis/lag_data/l7/',
          'band_index': 17,
          'band_name': 'dif_vi57',
          'MIN_YEAR_7': 2013,
          'LAG_YEAR': 3,
          'operation': 'dif',
          'satellite': 7}

    l5_1 = {'in_path': 'D:/svi_stack/Landsat_5/',
          'out_path': 'C:/Users/eduroscha001/Documents/thesis/lag_data/l5/',
          'band_index': 7,
          'band_name': 'dif_nbr2',
          'MIN_YEAR_5': 2008,
          'LAG_YEAR': 3,
          'operation': 'dif',
          'satellite': 5}

    l5_2 = {'in_path': 'D:/svi_stack/Landsat_5/',
          'out_path': 'C:/Users/eduroscha001/Documents/thesis/lag_data/l5/',
          'band_index': 17,
          'band_name': 'dif_vi57',
          'MIN_YEAR_5': 2002,
          'LAG_YEAR': 3,
          'operation': 'dif',
          'satellite': 5}

    #This calls which Landsats to compare
    parallel_raster([l7_1, l7_2, l5_2])

    totalEndTime = time.time() - allTimeStart
    total_minutes = float(totalEndTime) / 60
    print('This process took {} minutes, AKA {} seconds'.format(total_minutes, totalEndTime))

