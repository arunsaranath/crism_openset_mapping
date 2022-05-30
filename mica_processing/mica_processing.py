# -*- coding: utf-8 -*-

"""
FileName:               mica_processing
Author Name:            Arun M Saranathan
Description:            This code file is used to extract the numerators corresponding to the MICA spectra from the
                        images and place them in an SLI file

Dependancies:           spectral

Date Created:           22nd June 2021
Last Modified:          22nd June 2021
"""

'Import core packages'
import numpy as np
import os
import pandas as pd
from spectral.io.envi import SpectralLibrary as sli

from spectral_utilities.spectral_utilities import spectral_utilities

class mica_processing(object):
    def lbl_read(self, fName):
        """
        Define a function to read the label files associated with the MICA spectra

        :param fName: (string)
        The name of the label file to read

        :return: [ndarray]
        A numpy array which contains fields with the data from the label file
        """
        'Read a lbl file to a list'
        lines = open(fName, 'r').readlines()

        'Check if the files are empty'
        if (len(lines) == 0):
            raise ValueError('This lbl file is empty')

        'Extract and split the string pairs'
        lbl = [x.split('=') for x in lines]
        fields = []
        labels = []

        'look at all the labels'
        while (len(lbl) > 0):
            temp = lbl.pop(0)
            if (len(temp) == 2):
                field, label = temp
                labels.append(label.strip())
                fields.append(field.strip())

        return (np.vstack((np.asarray(fields), np.asarray(labels)))).T

    def mica_read(self, fName):
        """
        Define a function to  read the MICA file Data

        :param fName: (string)
        The name of the label file to read

        :return: (ndarray: [nBands])
        A numpy array which contains the MICA spectral data
        """
        'Read a lbl file to a list'
        lines = open(fName, 'r').readlines()

        'Check if the files are empty'
        if (len(lines) == 0):
            raise ValueError('This .tab file is empty')

        'Extract and split the string pairs'
        spectra_info = [x.split(',') for x in lines]
        spectra_info = np.array(spectra_info, dtype=np.float)

        'Replace dont care values by NaNs'
        a = np.where(spectraInfo == 65535)
        spectra_info[a] = np.nan

        return spectra_info

    def relab_read(self, fname, samp_cat_loc = '/Volume2/arunFiles/RelabDB2017Dec31/catalogues/Spectra_Catalogue.xls'):

        """
        Get the library spectra from the relab database

        :param fname: (string)
        The name of the label file to read

        :param samp_cat_loc: (string) (Default: '/Volume2/arunFiles/RelabDB2017Dec31/catalogues/Spectra_Catalogue.xls')
        The location where the RELAB Spectral Data is present. Default is set to the location on my local computer.

        :return: A numpy array which contains the RELAB library data
        """

        'Get the RELAB catalogue and see the spectrum'

        sample_catalogue = pd.ExcelFile(samp_cat_loc)
        sample_catalogue = sample_catalogue.parse(0)

        'find the row with the required information'
        chosen_row = sample_catalogue.loc[sample_catalogue['SpectrumID'] == fname]

        'Get the sample information'
        sample_info = ((chosen_row['SampleID']).iloc[0]).lower()
        sample_info = sample_info.split('-')

        'Get the file'
        base_str = '/Volume2/arunFiles/RelabDB2017Dec31/data/'
        add_str = base_str + str(sample_info[1]) + '/' + str(sample_info[0] + '/')
        add_str = add_str + fname.lower() + '.txt'

        'Read a RELAB file to a list'
        lines = open(add_str, 'r').readlines()[2::]

        'Extract and split the string pairs'
        spectra_info = [x.split() for x in lines]
        spectra_info = np.array(spectra_info, dtype=np.float)

        return spectra_info

    def usgs_read(self, fName, loc='/Volume2/arunFiles/MICA Library/ASCII/M/'):
        """

        :param fName: (string)
        The name of the label file to read.

        :param usgsLoc: (string) (Default: '/Volume2/arunFiles/MICA Library/ASCII/M/')
        The location where the USGS folder. By default set to the location on my local computer

        :return: [ndarray: nBands X 2]
        A numpy array which contains the USGS library data.
        """
        usgs_loc = os.listdir(loc)
        sample_list = [my_str for my_str in usgs_loc if fName in my_str]

        add_str = '/Volume2/arunFiles/MICA Library/ASCII/M/' + str(sample_list[-1])

        'Read a RELAB file to a list'
        lines = open(add_str, 'r').readlines()[17::]

        'Extract and split the string pairs'
        spectra_info = [x.split() for x in lines]
        spectra_info = np.array(spectra_info, dtype=np.float)

        'Replace dont care values by NaNs'
        a = np.where(spectra_info == -12300000000000000425850770517131264.0)
        spectra_info[a] = np.nan

        return spectra_info

    def extract_mica_numerators(self, mica_lblloc, mica_imgloc, save_loc=None):
        """
        This function can be used to extract the CRISM MICA numerators and place them in an ENVI Spectral library
        (.SLI) file.

        :param mica_lblloc: (string)
        The location where the MICA labels files are present. These files contain information about each numerator and
        the function will iterate through all these label files and extract the appropriate numerators.

        :param mica_imgloc: (string)
        The location where the actual CRISM images from which the MICA detections are made are present. Based on the
        information in the label files the appropriate numerators are extracted.

        :param save_loc: (string) (Default: None)
        If provided the function also saves the SLI at the location provided by the user.

        :return: spectral SLI object
        """

        assert os.path.isdir(mica_lblloc), "The label location does not exist"
        assert os.path.isdir(mica_imgloc), "The location with the MICA images does not exist"
        assert (save_loc is None) or isinstance(save_loc, str), \
            "The location to save the SLI must be an valid address string"

        'Initialize the variables to hold the data'
        name_endmem = []
        mica_endmem_spectra = []

        for r, d, f in os.walk(mica_lblloc):
            for file in f:
                'Image files will have the end ".lbl"'
                if file.find('.lbl') != -1:
                    'Get and read the label file'
                    lbl_file = os.path.join(r, file)
                    lbl_info = self.lbl_read(lbl_file)

                    'Extract the image name from the label'
                    img_name = lbl_info[0:116][19][1][1:12]
                    """img_name = img_name[1]
                    img_name = img_name[1:12]"""

                    col_num = int(lbl_info[0:116][20][1][:-8])
                    """col_num = col_num[1]
                    col_num = int(col_num[:-8])"""

                    row_num = int(lbl_info[0:116][21][1][:-8])
                    """row_num = row_num[1]
                    row_num = int(row_num[:-8])"""

                    'Extract ROI information'
                    roi_size = lbl_info[0:116][28][1]
                    xidx = roi_size.find('x')
                    roi_size = int((int(roi_size[1:xidx]) - 1) / 2)
                    #roi_size = (roi_size - 1) / 2

                    img_folder = os.path.join(mica_imgloc, img_name)

                    if (os.path.isdir(img_folder)):
                        for _, _, f1 in os.walk(img_folder):
                            for file1 in f1:
                                if file1.find('_nr_ds.img') != -1:
                                    'Get the image name and read the image'
                                    img_loc = os.path.join(img_folder, file1)
                                    cube, header = spectral_utilities().open_hsiImage(img_loc)

                                    'Get the ROI from the image'
                                    roi = np.asarray(cube[(row_num - roi_size):(row_num + roi_size + 1),
                                                     (col_num - roi_size):(col_num + roi_size + 1),
                                                     :])
                                    roi = roi.transpose(2, 0, 1).reshape(roi.shape[2], -1)

                                    spectra = np.squeeze(np.mean(roi, 1))

                                    name_endmem.append(lbl_info[30, 1][1:-1])
                                    mica_endmem_spectra.append(spectra)


        'Convert to numpy arrays'
        mica_endmem_spectra = np.asarray(mica_endmem_spectra)

        'Modify the header to suit the SLI'
        'Header details'
        sli_hdr = {'wavelength':np.asarray(header['wavelength'], dtype=np.float32),
                   'lines':mica_endmem_spectra.shape[0], 'samples': mica_endmem_spectra.shape[1], 'bands':1,
                   'spectra names':name_endmem}

        '.sli with MICA numerators'
        mica_num_sli = sli(mica_endmem_spectra, sli_hdr, [])
        'If needed save the SLI'
        if save_loc is not None:
            'If the directory mentioned in save does not exist make that directory'
            if not os.path.isdir(os.path.dirname(save_loc)):
                os.makedirs(os.path.dirname(save_loc))

            'Save the SLI'
            mica_num_sli.save(save_loc)

        return mica_num_sli


if __name__ == "__main__":
    'Create the MICA SLI'
    lbl_loc = '/Volume2/arunFiles/MICA Library/mrocr_8001/data/'
    mica_loc = '/Volume1/data/CRISM/atmCorr_stable/mica_images'

    save_loc = './data_products/mica_num_yukiProc'

    mica_num_sli = mica_processing().extract_mica_numerators(lbl_loc, mica_loc, save_loc=save_loc)



