# This script makes the input raster stack that is to be used for the ML algorithm

"""
Notes for the script
Conseptual Model:
takes the rasters from the lag, and takes selected bands from the other input rasters and makes it into a single raster stack
Sample naming convention for the SVI Stack post fire
LT05_CU_004013_20111001.tif
LE07_CU_004013_20020124.tif
LC08_CU_004013_20130324.tif
012345678901234567890123456
000000000011111111112222222
index above
Post fire Raster Stack band index
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
Pre Fire output band name example (all indices will have the same length of characters)
LT05_dif_nbr2_20060120.tif
LE07_dif_nbr2_20050116.tif
LC08_dif_nbr2_20170102.tif
01234567890123456789012345
00000000001111111111222222
index above
The two raster stacks I will need and the order of the bands
mix of agg and not agg for 20% sampled importance
Stack 1
SAVI    [12]
NDVI    [10]
GEMI    [4]
NDWI    [11]
dNBR2   (pre)
NBR2    [7]
dVI57   (pre)
VI57    [17]
mix of agg and not agg for 20% sampled forward
Stack 2
NDMI_AVG    (pre)
dNBR2       (pre)
GEMI_AVG    (pre)
VI46_AVG    (pre)
VI6T_STD    (pre)
GEMI        [4]
dVI57       (pre)
NDWI        [11]
The different file paths
For the post fire data:
F:/Thesis/svi_stack/Landsat_8/
For the pre fire data:
//geo-procella/GradData/cross8046/l8_lag_single_band_indices/
"""

# Import the needed modules
try:
    from osgeo import gdal
    from osgeo import gdalconst
except ImportError as e:
    print(e)
    import gdal
    import gdalconst
import numpy as np
import libs.indices as ind
import time, rasterio, os, re, time


# =======================================================================================================================
#                   Helping Functions
# =======================================================================================================================


def svi_file_search(root_path, regex_name):
    """
    this gets the right file for the stack from the loose raster bands
    :param root_path: thefolder where the svis live, string
    :param regex_name: the search name in xxxx_yyyymmdd format. ex: gemi_20170602
    :return: the correct raster fp
    """
    # the input name should be xxxx_yyyymmdd where x = svi and then the date, should only be one for the stak matching both things

    # get the regex thing from the regex search term inputted into the function
    regex = re.compile(regex_name)

    for file in os.listdir(root_path):
        if file.endswith('.tif'):
            if regex.match(file):
                full_file_path = os.path.join(root_path, file)

                dataset = ind.read_data(full_file_path, raster_driver_name='GTiff')

                cols = dataset.RasterXSize
                rows = dataset.RasterYSize
                projection = dataset.GetProjection()
                metadata = dataset.GetMetadata()
                geotransform = dataset.GetGeoTransform()

                # get the band
                band = dataset.GetRasterBand(1)
                # make an array
                array = band.ReadAsArray()
                # make the array have no nan values
                #print('single test 1: ', np.any(np.isnan(array)))
                nan_idx = np.isnan(array)
                array[nan_idx] = -9999
                #print('single test 2: ', np.any(np.isnan(array)))

                # make an output path
                new_file_path = os.path.join('C:/tmp/', file)

                # write to a temporary folder
                ind.output_single_band_raster(data=array, out_fp=new_file_path, col_size=cols, row_size=rows,
                                              num_band=1,
                                              raster_driver_name='GTiff', projection=projection,
                                              geotransform=geotransform, metadata=metadata, nodataval=-9999)
                return new_file_path


# This was taken from the indices module and manipulated. It uses rasterio
def create_raster(file_list, out_path, name):
    """
    :return: A raster that combines all of the added rasters
    """

    # looks like I will have to make a temp folder with the SVIs then add them all to a new raster then delete
    #   the temp SVIs

    # write the SVI as a band with rasterio
    # https://gis.stackexchange.com/questions/223910/using-rasterio-or-gdal-to-stack-multiple-bands-without-using-subprocess-commands
    # https://gis.stackexchange.com/questions/49706/adding-band-to-existing-geotiff-using-gdal

    # empyt list of the files
    # file_list = []

    # print('create raster list: ', file_list)

    # Read metadata of first file
    with rasterio.open(file_list[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count=len(file_list))

    # out file
    file_out = os.path.join(out_path + name)

    # Read each layer and write it to stack
    with rasterio.open(file_out, 'w', **meta) as dst:
        for id, layer in enumerate(file_list, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))


def delete_temp(temp_folder):
    #delete the temporary SVI files in the temp folder
    for i in os.listdir(temp_folder):
        try:
            del_path = temp_folder + i
            os.remove(del_path)
        except OSError as e:
            print("Error: %s : %s" % (i, e.strerror))


# C:\Thesis\test_data\LC08_dif_nbr2_20170102.tif
# 0000000000111111111122222222223333333333444444
# 0123456789012345678901234567890123456789012345
# [25:33]
# [5:14]

# =======================================================================================================================
#                   Main Function
# =======================================================================================================================


def stack_maker(svi_in_folder, lag_in_folder, out_folder, out_identifier, satellite, lag_band, svi_band,
                file_type='.tif', temp_path='C:/tmp/'):
    """
    Input lag and svi .tif rasters to join to be used for a ML classifier
    :param svi_in_folder: The folder path for the SVI data; string
    :param lag_in_folder: The folder path for the Lag data; string
    :param out_folder: The output data folder; string
    :param out_identifier: The output name identifier, the format is satellite_outIdentifier_date.tif
    :param satellite: satellite number; integer; ex: 8
    :param file_type: type of raster extension; string; default '.tif'
    :param temp_path: file folder path for temporary files; string; default C:/tmp
    :param lag_band: strings for the names of the input bands; string; ex: 'dNBR'
    :param svi_band: Key-Word dictionary of the SVI and the band index wanted from the
    :return: An input file for the ML classifier. The order of the input will be the order of the lag, then the order
                of the multiband SVI
    """
    # helps with errors
    # try:
    if (1 == 1):
        # I want to see how long the process takes, this will mark the iterations
        iteration_num = 0

        if satellite == 8:
            sat_name = 'LC08'
        elif satellite == 7:
            sat_name = 'LE07'
        elif satellite == 5:
            sat_name = 'LT05'
        else:
            print('pick a different satellite, either 8, 7, or 5 as an integer')
            return None

        # ===============================================================================================================
        # read through the files in the directory
        for file in os.listdir(svi_in_folder):
            if file.endswith(file_type):
                # make the full path
                in_image = os.path.join(svi_in_folder, file)

                # I want to see how long the process takes
                start_time = time.time()

                # To be efficient, I will first check if the data exists for the prefire bands since there will be more
                #      postfire images then prefire images due to the shift thing

                # get the date of the post_img_date
                post_img_date = file[15:23]

                # make an empty list that the file paths will be combined from
                compile_paths_list = []

                # get the file paths for the single bands and append to the list
                if (len(lag_band) != 0):
                    for lag in lag_band:
                        regex_name = sat_name + '_' + lag + '_' + post_img_date + file_type
                        lag_out_fp = svi_file_search(lag_in_folder, regex_name)
                        compile_paths_list.append(lag_out_fp)

                    if (None in compile_paths_list):
                        continue

                # ======================================================================================================
                # Add the bands from the multiband_svi images
                # output single bands to the tmp folder, then combine them later

                for key, value in svi_band.items():
                    # temporary output name and path for each svi
                    svi_out_name = sat_name + '_pst_' + key + '_' + post_img_date + file_type
                    svi_out_fp = os.path.join(temp_path, svi_out_name)
                    # Python version of Dan's method using GDAL
                    # make the options for GDAL
                    if (file_type == '.tif'):
                        opts = '-b ' + str(value) + ' -of GTiff'
                    else:
                        opts = '-b ' + str(value)
                    # open the raster multiband image
                    src_ds = gdal.Open(in_image)
                    # export the correct band to be a single band raster
                    ds = gdal.Translate(svi_out_fp, src_ds, options=opts)
                    # make the datasets none to save memory
                    ds = None
                    src_ds = None
                    # add the filepath to the list
                    compile_paths_list.append(svi_out_fp)

                # ======================================================================================================
                # Time to make into a single input raster stack for the ML classifier
                # print the list to make sure it worked
                print('Input SVI File order: \n{}'.format(compile_paths_list))

                # Make an output name
                output_name = sat_name + out_identifier + post_img_date + file_type

                # Output the image
                create_raster(compile_paths_list, out_folder, output_name)

                # Delete the data from the temp folder
                delete_temp(temp_path)

                # I want to see how long the process takes
                time_dif = time.time() - start_time
                iteration_num += 1
                print('Iteration {} took {} seconds to complete'.format(iteration_num, time_dif))

    # except Exception as e:
    #     print(e)

# =======================================================================================================================
#                   Run the script
# =======================================================================================================================

if __name__ == '__main__':
    # Landsat 8
    # stack_maker_a('F:/Thesis/svi_stack/Landsat_8/', 'F:/Thesis/input_stack/a/', 8)

    lag_band_a = ['dif_nbr2', 'dif_vi57']
    svi_band_a = {'SAVI': 12, 'NDVI': 10, 'GEMI': 4, 'NBR2 ': 7, 'VI57': 17}
    lag_band_b = ['avg_ndmi', 'dif_nbr2', 'avg_gemi', 'avg_vi46', 'std_vi6t', 'dif_vi57']
    svi_band_b = {'GEMI': 4, 'NDWI': 11}
    lag_band_c = ['dif_nbr2', 'dif_vi57']
    svi_band_c = {'GEMI': 4, 'NBR2 ': 7, 'NDWI': 11, 'VI57': 17}
    lag_band_d = ['dif_nbr2', 'dif_vi57']
    svi_band_d = {'SAVI': 12, 'NBR2 ': 7, 'NDWI': 11, 'VI57': 17}



    # Landsat 8 Stack D
    # stack_maker(svi_in_folder='D:/svi_stack/Landsat_8/',
    #             lag_in_folder='C:/Users/eduroscha001/Documents/thesis/lag_data/single_year_lag/l8/',
    #             out_folder='../input_stack/d/single_year_lag/',
    #             out_identifier='_stack_d_',
    #             satellite=8,
    #             lag_band=lag_band_d,
    #             svi_band=svi_band_d,
    #             file_type='.tif',
    #             temp_path='C:/tmp/')

    # Landsat 7 Stack D
    # stack_maker(svi_in_folder='D:/svi_stack/Landsat_7/',
    #             lag_in_folder='C:/Users/eduroscha001/Documents/thesis/lag_data/single_year_lag/l7/',
    #             out_folder='../input_stack/d/single_year_lag/',
    #             out_identifier='_stack_d_',
    #             satellite=7,
    #             lag_band=lag_band_d,
    #             svi_band=svi_band_d,
    #             file_type='.tif',
    #             temp_path='C:/tmp/')

    # Landsat 5 Stack D
    stack_maker(svi_in_folder= 'D:/svi_stack/Landsat_5/',
            lag_in_folder= 'C:/Users/eduroscha001/Documents/thesis/lag_data/single_year_lag/l5/',
            out_folder= '../input_stack/d/single_year_lag/',
            out_identifier = '_stack_d_',
            satellite = 5,
            lag_band = lag_band_d,
            svi_band = svi_band_d,
            file_type = '.tif',
            temp_path='C:/tmp/')