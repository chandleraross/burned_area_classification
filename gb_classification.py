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

#I want to see how long the process takes
start_time = time.time()


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
    # try:
    if(1==1):

        #call the Model
        with open(model, 'rb') as model_file:
            model_gbm = pickle.load(model_file)

        # I want to see how long the process takes, this will mark the iterations
        iteration_num = 0

        print('Ignore the error: "X does not have valid feature names, but"')

        #read through the files in the directory
        for file in os.listdir(in_path):
            if file.endswith('.tif'):

                # make the full path
                in_image = in_path + file

                # Change any NaN to -9999
                nan_fix(in_image, band_count)


                #read in the raster
                r_preds = pml.Raster(in_image)

                #Make the prediction name
                # pred_name1 = file[:-4] + '_1'
                # pred_name2 = file[:-4] + '_2'
                # pred_name3 = file[:-4] + '_3'
                # pred_name4 = file[:-4] + '_4'
                # pred_name5 = file[:-4] + '_5'
                # pred_name6 = file[:-4] + '_6'
                # pred_name7 = file[:-4] + '_7'
                # pred_name8 = file[:-4] + '_8'

                print(r_preds.names)
                #Change the prediction names
                # r_preds.rename({pred_name1:"SAVI",
                #                pred_name2:"NDVI",
                #                pred_name3:"GEMI",
                #                pred_name4:"NDWI",
                #                pred_name5:"dNBR2",
                #                pred_name6:"NBR2",
                #                pred_name7:"dVI57",
                #                pred_name8:"VI57"})

                if(stack=='stack_a'):
                    renamed = r_preds.rename(dict(zip(r_preds.names, ["dNBR2", "dVI57", "SAVI", "NDVI", "GEMI", "NBR2", "VI57"])))
                elif(stack=='stack_b'):
                    renamed = r_preds.rename(dict(zip(r_preds.names, ["SAVI", "NDVI", "GEMI", "NDWI", "dNBR2", "NBR2", "dVI57", "VI57"])))
                else:
                    print('Choose either "stack_a" or "stack_b"')

                print(renamed.names)

                # stack a
                # r_preds.rename({pred_name1:"dNBR2",
                #                pred_name2:"dVI57",
                #                pred_name3:"SAVI",
                #                pred_name4:"NDVI",
                #                pred_name5:"GEMI",
                #                pred_name6:"NBR2",
                #                pred_name7:"VI57"})

                # Stack B
                # r_preds.rename({pred_name1:"NDMI_AVG",
                #                pred_name2:"dNBR2",
                #                pred_name3:"GEMI_AVG",
                #                pred_name4:"VI46_AVG",
                #                pred_name5:"VI6T_STD",
                #                pred_name6:"dVI57",
                #                pred_name7:"GEMI",
                #                pred_name8:"NDWI"})

                #classify the raster image
                # result = r_preds.predict_proba(estimator=model_gbm)
                result = renamed.predict_proba(estimator=model_gbm)

                #write the output
                # classified_output = out_path + file[:-4] + '_t20_' + 'classified.tif'
                out_file = file[:-4] + threshold_name + 'classified.tif'
                classified_output = os.path.join(out_path, out_file)
                result.write(classified_output)

            # I want to see how long the process takes
            time_dif = time.time() - start_time
            minutes = float(time_dif)/60
            iteration_num += 1
            print('Iteration {} took {} seconds ({} minutes) to complete'.format(iteration_num, time_dif, minutes))

    # except Exception as e:
    #     print(e)

#email to let me know the script ran and completed while I was away
def email_me(sender, receiver):
    # Import smtplib for the actual sending function
    import smtplib
    # Import the email modules we'll need
    from email.message import EmailMessage

    # Create a text/plain message
    msg = EmailMessage()
    msg.set_content('The Cookies are Ready')

    # me = 'cr.python.dev@gmail.com'
    # you = 'cross8046@sdsu.edu'
    msg['Subject'] = 'Python Script Status'
    msg['From'] = sender
    msg['To'] = receiver

    # Send the message via our own SMTP server.
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()

if __name__ == '__main__':
    main(in_path='C:/Users/eduroscha001/Documents/thesis/input_stack/b/',
         # in_path='D:/input_stack/b/l5/',
         # in_path='C:/Users/Chandler/Documents/thesis_scripts/BurnedAreaClassification/input_stack/b/',
         # in_path='D:/input_stack/a/classified_at_school/',
         out_path='C:/Users/eduroscha001/Documents/thesis/classified_maps/b/',
         # out_path='C:/tmp/classified_maps/',
         model='./models/3.9/stack_b_thresh_50.pkl',
         # model='',
         threshold_name='_t50_',
         stack='stack_b',
         band_count=8)
    print('All rasters are classified!')
    #send an email once complete
    #email_me(sender='cr.python.dev@gmail.com', receiver='cross8046@sdsu.edu')

#Helpful sources
#http://www.wvview.org/open_source_gis/site_renders/Spatial_ML/site/index.html#predict-to-raster-data