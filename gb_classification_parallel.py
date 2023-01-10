#Takes the predictores and outputs a classified burned/not burned map
#Chandler Ross
#1/27/2022

#=======================================================================================================================

#Import the necessary libraries
import pickle, time, os
import pyspatialml as pml
import libs.indices as ind
import numpy as np
try:
    from osgeo import gdal
    from osgeo import gdal_array
    from osgeo import gdalconst
except ImportError as e:
    print(e)
    import gdal
    import gdal_array
    import gdalconst

from multiprocessing import Process, managers, cpu_count, current_process


#=======================================================================================================================
#                   Step 1: Call the model
#=======================================================================================================================

'''
Pickle Model names
stack_a_thresh_20.pkl
stack_a_thresh_50.pkl
stack_a_thresh_80.pkl
stack_b_thresh_20.pkl
stack_b_thresh_50.pkl
stack_b_thresh_80.pkl
'''


#Function to fix the NaN to -9999 issue
def nan_fix(fp, band_count=7):
    #while loop to fix each band individually
    b = 1
    while b < band_count+1:
        #read the fp as a dataset
        dataset = ind.read_data(fp)
        #get band count
        if(b==1):
            print('# of Bands: {}'.format(dataset.RasterCount))
        #call the band
        band = dataset.GetRasterBand(b)
        #turn into an array
        array = band.ReadAsArray()
        #get the na value
        nan_idx = np.isnan(array)
        #change nan value iin the array to -9999
        array[nan_idx] = -9999
        #get the band type
        print(gdal.GetDataTypeName(band.DataType))

        print('Any NaN Values:        {}'.format(np.any(np.isnan(array))))
        print('Any Infinite Values:   {}'.format(np.any(np.isinf(array))))
        # print('All is finite:         {}'.format(np.all(np.isfinite(array))))
        # sorted_array = np.sort(array)
        # print('Smallest Value:        {}'.format(sorted_array[0:1]))
        # print('Largest Value:         {}'.format(sorted_array[-2:-1]))
        #make the loop go up by one
        b += 1


#=======================================================================================================================
#                   Step 2: Classify the rasters
#=======================================================================================================================


def main(in_path, out_path, model, threshold_name, stack, band_count):
    try:
    # if(1==1):

        #call the Model
        with open(model, 'rb') as model_file:
            model_gbm = pickle.load(model_file)

        # I want to see how long the process takes, this will mark the iterations
        iteration_num = 0

        print('Ignore the error: "X does not have valid feature names, but"')

        #read through the files in the directory
        for file in os.listdir(in_path):
            if file.endswith('.tif'):
                # I want to see how long the process takes
                # start_time = time.time()

                # make the full path
                in_image = in_path + file

                # Change any NaN to -9999
                nan_fix(in_image, band_count)


                #read in the raster
                r_preds = pml.Raster(in_image)

                print(r_preds.names)

                #Change the prediction names

                if(stack=='stack_a'):
                    renamed = r_preds.rename(dict(zip(r_preds.names, ["dNBR2", "dVI57", "SAVI", "NDVI", "GEMI", "NBR2", "VI57"])))
                elif(stack=='stack_b'):
                    renamed = r_preds.rename(dict(zip(r_preds.names, ["SAVI", "NDVI", "GEMI", "NDWI", "dNBR2", "NBR2", "dVI57", "VI57"])))
                elif(stack=='stack_c'):
                    renamed = r_preds.rename(dict(zip(r_preds.names, ["dNBR2", "dVI57", "GEMI", "NBR2", "NDWI", "VI57"])))
                elif (stack == 'stack_d'):
                    renamed = r_preds.rename(dict(zip(r_preds.names, ["dNBR2", "dVI57", "SAVI", "NBR2", "NDWI", "VI57"])))
                else:
                    print('Choose either "stack_a" or "stack_b"')

                print(renamed.names)


                #classify the raster image
                result = renamed.predict_proba(estimator=model_gbm)

                #write the output
                out_file = file[:-4] + threshold_name + 'classified.tif'
                classified_output = os.path.join(out_path, out_file)
                result.write(classified_output)

            # I want to see how long the process takes
            # time_dif = time.time() - start_time
            # minutes = float(time_dif)/60
            iteration_num += 1
            # print('Iteration {} took {} seconds ({} minutes) to complete'.format(iteration_num, time_dif, minutes))
            print('I used stack: {} and threshold: {} for this classification'.format(stack, threshold_name))

    except Exception as e:
        print(e)


"""
if __name__ == '__main__':
    main(#in_path='C:/Users/eduroscha001/Documents/thesis/input_stack/a/',
         # in_path='D:/input_stack/b/l5/',
         in_path='C:/Users/Chandler/Documents/thesis_scripts/BurnedAreaClassification/input_stack/b/',
         # in_path='D:/input_stack/a/classified_at_school/',
         # out_path='C:/Users/eduroscha001/Documents/thesis/classified_maps/a/',
         out_path='C:/tmp/classified_maps/',
         model='./models/3.9/stack_b_thresh_80.pkl',
         threshold_name='_t80_',
         stack='stack_b',
         band_count=8)
    print('All rasters are classified!)
"""

#Helpful sources
#http://www.wvview.org/open_source_gis/site_renders/Spatial_ML/site/index.html#predict-to-raster-data


#=======================================================================================================================
#                   Parallization
#=======================================================================================================================

def combined_function(d):

    if(d['stack']=='stack_a' and d['threshold_name']=='_t20_'):
        main(d['in_path'], d['out_path'], d['model'], d['threshold_name'], d['stack'], d['band_count'])

    elif(d['stack']=='stack_a' and d['threshold_name']=='_t50_'):
        main(d['in_path'], d['out_path'], d['model'], d['threshold_name'], d['stack'], d['band_count'])

    elif(d['stack']=='stack_a' and d['threshold_name']=='_t80_'):
        main(d['in_path'], d['out_path'], d['model'], d['threshold_name'], d['stack'], d['band_count'])

    elif(d['stack']=='stack_b' and d['threshold_name']=='_t20_'):
        main(d['in_path'], d['out_path'], d['model'], d['threshold_name'], d['stack'], d['band_count'])

    elif(d['stack']=='stack_b' and d['threshold_name']=='_t50_'):
        main(d['in_path'], d['out_path'], d['model'], d['threshold_name'], d['stack'], d['band_count'])

    elif(d['stack']=='stack_b' and d['threshold_name']=='_t80_'):
        main(d['in_path'], d['out_path'], d['model'], d['threshold_name'], d['stack'], d['band_count'])

    elif(d['stack']=='stack_c' and d['threshold_name']=='_t20_'):
        main(d['in_path'], d['out_path'], d['model'], d['threshold_name'], d['stack'], d['band_count'])

    elif(d['stack']=='stack_c' and d['threshold_name']=='_t50_'):
        main(d['in_path'], d['out_path'], d['model'], d['threshold_name'], d['stack'], d['band_count'])

    elif(d['stack']=='stack_c' and d['threshold_name']=='_t80_'):
        main(d['in_path'], d['out_path'], d['model'], d['threshold_name'], d['stack'], d['band_count'])

    elif(d['stack']=='stack_d' and d['threshold_name']=='_t20_'):
        main(d['in_path'], d['out_path'], d['model'], d['threshold_name'], d['stack'], d['band_count'])

    elif(d['stack']=='stack_d' and d['threshold_name']=='_t50_'):
        main(d['in_path'], d['out_path'], d['model'], d['threshold_name'], d['stack'], d['band_count'])

    elif(d['stack']=='stack_d' and d['threshold_name']=='_t80_'):
        main(d['in_path'], d['out_path'], d['model'], d['threshold_name'], d['stack'], d['band_count'])

    else:
        print('This should not be printed, fix the dictionaries or something')


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


#=================================================
# Run the stuff
#=================================================

in_a = 'C:/Users/eduroscha001/Documents/thesis/input_stack/a/'
in_b = 'C:/Users/eduroscha001/Documents/thesis/input_stack/b/'
in_c = 'C:/Users/eduroscha001/Documents/thesis/input_stack/c/'
in_d = 'C:/Users/eduroscha001/Documents/thesis/input_stack/d/'

# Run this function
if __name__ == '__main__':
    d1 = {#'in_path': 'C:/Users/eduroscha001/Documents/thesis/input_stack/d/img1/',
          'in_path': 'C:/Users/eduroscha001/Documents/thesis/input_stack/d/single_year_lag/1/',
          'out_path': 'D:/classified_maps/d/single_year_lag/t20/',
          'model': './models/stack_d_thresh_20_30agg_even_dropped.pkl',
          'threshold_name': '_t20_',
          'stack': 'stack_d',
          'band_count': 6}

    d2 = {#'in_path': 'C:/Users/eduroscha001/Documents/thesis/input_stack/c/img2/',
          'in_path': 'C:/Users/eduroscha001/Documents/thesis/input_stack/d/single_year_lag/2/',
          'out_path': 'D:/classified_maps/d/single_year_lag/t20/',
          'model': './models/stack_d_thresh_20_30agg_even_dropped.pkl',
          'threshold_name': '_t20_',
          'stack': 'stack_d',
          'band_count': 6}

    d3 = {#'in_path': 'C:/Users/eduroscha001/Documents/thesis/input_stack/c/img3/',
          'in_path': 'C:/Users/eduroscha001/Documents/thesis/input_stack/d/single_year_lag/3/',
          'out_path': 'D:/classified_maps/d/single_year_lag/t20/',
          'model': './models/stack_d_thresh_20_30agg_even_dropped.pkl',
          'threshold_name': '_t20_',
          'stack': 'stack_d',
          'band_count': 6}

    d4 = {#'in_path': 'C:/Users/Chandler/Documents/thesis_scripts/BurnedAreaClassification/input_stack/a/',
          'in_path': 'C:/Users/eduroscha001/Documents/thesis/input_stack/d/1/',
          'out_path': 'D:/classified_maps/d/dropped/t20/',
          'model': './models/stack_d_thresh_20_30agg_even_dropped.pkl',
          'threshold_name': '_t20_',
          'stack': 'stack_d',
          'band_count': 6}


    d5 = {#'in_path': 'C:/Users/Chandler/Documents/thesis_scripts/BurnedAreaClassification/input_stack/b/',
          'in_path': 'C:/Users/eduroscha001/Documents/thesis/input_stack/d/single_year_lag/4/',
          'out_path': 'D:/classified_maps/d/single_year_lag/t20/',
          'model': './models/stack_d_thresh_20_30agg_even_dropped.pkl',
          'threshold_name': '_t20_',
          'stack': 'stack_d',
          'band_count': 6}

    d6 = {#'in_path': 'C:/Users/Chandler/Documents/thesis_scripts/BurnedAreaClassification/input_stack/b/',
          'in_path': in_b,
          'out_path': 'C:/Users/eduroscha001/Documents/thesis/classified_maps/b/t80/',
          'model': './models/3.9/stack_a_thresh_80.pkl',
          'threshold_name': '_t80_',
          'stack': 'stack_a',
          'band_count': 7}

    #This calls which ones to call
    parallel_raster([d1, d2, d3])