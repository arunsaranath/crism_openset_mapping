# -*- coding: utf-8 -*-

"""
FileName:               crism_reg_sigSamp
Author Name:            Arun M Saranathan
Description:            This code file is used to significant samples from CRISM images in a specific region and store
                        the samples from that region in a specific pandas HDFS store. The data can then be used for test
                        ing/training vaious models

Dependancies:           extract_samples

Date Created:           21st June 2021
Last Modified:          21st June 2021
"""

'Import core packages'
import pandas as pd
import os

'Import user defined package'
from extract_samples import extract_samples

class create_sigSamp_store(object):
    def __init__(self, save_loc=None):
        """
        This class is used extract spectra with significant absorptions for a set of CRISM images. The functions in this
        class will iterate over all the images in specific folder and extract significant samples from the images which
        contain a specific search string.

        :param save_loc: (string) (Default: None)
        A physical address where the Pandas HDFStore with all the signigicant spectral absorptions are stored. If none
        is provided it is saved in the 'data_stores' subfolder of the present folder.
        """

        print("Iterating over a set of CRISM images to extract spectra with significant absorption features\n")

        'If not provided make create an appropriate location to save the data'
        if save_loc is None:
            save_loc = os.path.join(os.getcwd(), 'data_stores')

        'Check the parameters provided'
        assert isinstance(save_loc, str), "The save_loc must be a string where you want to save the significant samples"

        'Now update the variables of the object'
        self.__save_loc = save_loc

        'Additional variables to track/save spectra with significant absorptions'
        self.img_info_df = pd.DataFrame([])
        self.spectral_data_df = pd.DataFrame([])
        self.cospectral_data_df = pd.DataFrame([])
        self.wavelength_df = pd.DataFrame([])
        self.region_name = None
        'Create a variable to track the number of images processed'
        self.__nimages = 0

    def process_folder(self, region_loc, img_string=".img", region_name=None):
        """
        This function can be used to iterate over all the files inside the folder location provided and extracts the
        spectral samples with significant absorptions from the images of the appropriate type.

        :param region_loc: (string)
        The string which is the address of a physical location which contains a bunch of images of the type we want to
        process.

        :param img_string (string) (Default:None)
        This argument provides an additional mode of filtering the CRISM images in the specific folder. In this case it
        only considers image which consists of the sting provided in this argument.

        :param region_name(string)(Default: None)
        The name of the region, if the data is saved it is saved in a HDFSFile with the region_name

        :return: None
        """
        'Check the parameters provided'
        assert isinstance(region_loc, str), "The folder location which contains the images should be string"
        assert os.path.isdir(region_loc), "The folder provided by the user does not exist"
        assert isinstance(img_string, str), "The img_string parameter must be a string"

        'Get the region name'
        region_name = os.path.basename(region_loc)
        if self.region_name is None:
            self.region_name = region_name
        else:
            self.region_name += ('_' + region_name)

        'Iterate over the files in the specified folder'
        for r, d, f in os.walk(region_loc):
            for file in f:
                if file.find(img_string) != -1:
                    'The image address is'
                    img_address = os.path.join(r, file)

                    'Get the info from a specific image'
                    df_info, df_spectra, df_cont_spectra = \
                        extract_samples(strtWvl=1.0275).extract_nonFlatSpectra(img_address)

                    'Add this data to the folder'
                    self.img_info_df = self.img_info_df.append(df_info)
                    self.spectral_data_df = self.spectral_data_df.append(df_spectra)
                    self.cospectral_data_df = self.cospectral_data_df.append(df_cont_spectra)

                    'Update count of number of images processed'
                    self.__nimages += 1

        print('In this search we have processed {:d} images'.format(self.__nimages))

    def save_store(self, store_name=None):
        """
        This function can be used to save data in HDFStore with multiple tables.

        :param store_name: (string)
        The name to be assigned to the pandas HDFStore when saving. If none is provided it will be the region name.

        :return:
        """
        if store_name is None:
            if self.region_name is not None:
                store_name = self.region_name
            else:
                store_name ='store'

        assert isinstance(store_name, str), "The store name has to be a string"


        'Check is the save location exists on the folder'
        if not os.path.isdir(self.__save_loc):
            'If not make the directory'
            os.mkdir(self.__save_loc)

        'Add store name to this'
        store_name = os.path.join(self.__save_loc, (store_name + '.h5'))

        'Create a pandas HDFStore'
        store = pd.HDFStore(store_name, "w")
        'Now save the data we have extracted'
        store.append('img_info', self.img_info_df, format='table', data_columns=True)
        store.append('spectral_data', self.spectral_data_df, format='table', data_columns=True)
        store.append('cospectral_data', self.cospectral_data_df, format='table', data_columns=True)

        store.close()


if __name__ == "__main__":
    'Provide the name of the location'
    folder_loc = '/Volume1/data/CRISM/atmCorr_stable/mica_images'
    'Create an object to make the appropriate store'
    obj1 = create_sigSamp_store()
    obj1.process_folder(folder_loc, img_string='_nr_ds.img')
    obj1.save_store()





