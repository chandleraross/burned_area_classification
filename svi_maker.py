"""
This script makes a raster stack of SVIs
Chandler Ross
"""

import libs.indices as ind
import rasterio, os, time
from matplotlib import pyplot as plt


def main(in_path, out_path):

    driver_name = 'GTiff'

    #helps with errors
    try:
        # I want to see how long the process takes, this will mark the iterations
        iteration_num = 0

        #read through the files in the directory
        for file in os.listdir(in_path):
            if file.endswith('.tif'):

                #make the full path
                in_image = in_path + file

                #I want to see how long the process takes
                start_time = time.time()

                #in_image = './data/practice/LE07_CU_004013_20000103.TIF.tif'

                #Compute the SVIs for the image
                ind.get_bai(in_image)
                ind.get_csi(in_image)
                ind.get_evi(in_image)
                ind.get_gemi(in_image)
                ind.get_miribi(in_image)
                ind.get_nbr(in_image)
                ind.get_nbr2(in_image)
                ind.get_nbrt1(in_image)
                ind.get_ndmi(in_image)
                ind.get_ndvi(in_image)
                ind.get_ndwi(in_image)
                ind.get_savi(in_image)
                ind.get_vi6t(in_image)
                ind.get_vi43(in_image)
                ind.get_vi45(in_image)
                ind.get_vi46(in_image)
                ind.get_vi57(in_image)

                #make a name for the output
                name_temp = str(file)
                out_file_name = name_temp[:23] + '.tif'

                #Take the SVIs and make them into a single raster
                ind.create_raster('./temp/', out_path, out_file_name)

                #delete the tempoary svis
                ind.delete_svis('./temp/')

                #I want to see how long the process takes
                time_dif = time.time() - start_time
                iteration_num += 1
                print('Iteration {} took {} seconds to complete'.format(iteration_num, time_dif))


    except Exception as e:
        print(e)


if __name__ == '__main__':
    main('./data/', './output/')
    print("completed!")
