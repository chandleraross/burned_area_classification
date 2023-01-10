#The goal of this script is to reclassify rasters faster than ArcGIS Can

#Going to read in the rasters as an array. them I will change the np array
# https://www.statology.org/numpy-replace/#:~:text=You%20can%20use%20the%20following%20methods%20to%20replace,of%2020%20my_array%20%5Bmy_array%20%3E%208%5D%20%3D%2020


# Import GDAL and other libraries
try:
    from osgeo import gdal
    from osgeo import gdalconst
except ImportError as e:
    print(e)
    import gdal
    import gdalconst



#Make a class that can read in rasters and turn it into an array

class RasterReader:
    def __init__(self, in_file, out_file, raster_driver_name='GTiff'):
        self.in_file = in_file
        self.out_file = out_file
        self.raster_driver_name = raster_driver_name

    def read_data(self, fp_in_img, raster_driver_name='GTiff'):
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

    def get_info(self, fp_in_img, fp_out_info, flag_print=False, raster_driver_name='GTiff'):
        """
        get input raster data information
        :param fp_in_img: input raster image file path (string)
        :param fp_out_info: ouput file path (string)
        :param flag_print: flag to print raster information (boolean)
        :param raster_driver_name: input raster driver name (string)
        :return: a dictionary containing raster information (dictionary)
        """
        path, filename = os.path.split(fp_in_img)

        dataset = __read_data(fp_in_img, raster_driver_name)

        str_out = ""
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        bands = dataset.RasterCount
        projection = dataset.GetProjection()
        metadata = dataset.GetMetadata()

        raster_info = {
            'file_path': fp_in_img,
            'file_name': filename,
            'cols': cols,
            'rows': rows,
            'bands': bands,
            'projection': projection,
            'metadata': metadata
        }

        str_out += "Input Image: '{}'\n".format(fp_in_img)
        str_out += "Num Columns: {}\nNum Rows: {}\nNum Bands: {}\nProjection: {}\n".format(cols,rows,bands,projection)

        geotransform = dataset.GetGeoTransform()
        if geotransform:
            origin_x = geotransform[0]
            origin_y = geotransform[3]
            pixel_width = geotransform[1]
            pixel_height = geotransform[5]

            str_out += "Geotransform:\n\torigin:({}, {})\n\tpixel width: {}\n\tpixel height: {}\n".format(origin_x,origin_y,geotransform[1],geotransform[5])

        else:
            str_out += "Geotransform: {}".format(geotransform)

        raster_info['geotransform'] = geotransform

        with open(fp_out_info, 'w') as out:
            out.write(str_out)

        # reading raster data for each band
        str_out += "\nRaster Band Information:\n"

        band_info = {}

        # iterate each band
        for i in range(bands):
            band = dataset.GetRasterBand(i+1)

            # cf) band data type & no data value
            band_type = gdal.GetDataTypeName(band.DataType)
            nodataval = band.GetNoDataValue()

            min = band.GetMinimum()
            max = band.GetMaximum()

            if min is None or max is None:
                min, max = band.ComputeRasterMinMax()

            str_out += "Band {}:\n\tBand DataType: {}\n\tMin Pixel Value: {}\n\tMax Pixel Value: {}\n".format(i+1, band_type, min, max)

            band_info[i+1] = {'band': i+1, 'band_type': band_type, 'nodata_val': nodataval, 'min': min, 'max':max}

            band = None
            data = None

        raster_info['band_info'] = band_info

        dataset = None

        if flag_print:
            print(str_out)

        with open(fp_out_info, 'w') as out:
            out.write(str_out)

        return raster_info

    def read_as_array(self):
        pass