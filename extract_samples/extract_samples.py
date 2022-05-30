# -*- coding: utf-8 -*-

"""
FileName:               extract_samples
Author Name:            Arun M Saranathan
Description:            This code file is used to extract samples the CRISM Images where the MICA samples were
                        originally found. A simple heurestic is used to eliminate spectra which appear generally flat so
                        that they do not affect the model training.

Dependancies:           spectral_utilities, crism_processing

Date Created:           19th June 2021
Last Modified:          20th June 2021
"""

'Import core packages'
import os
import numpy as np
import pandas as pd
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

'User Defined Packages'
from spectral_utilities.spectral_utilities import spectral_utilities
from crism_processing.crism_processing import crism_processing

class extract_samples(object):
    def __init__(self, strtWvl=1.02, stopWvl=2.6):
        """
        Constructor of the class used to extract non-flat spectra from CRISM L images in the Near-IR spectral range. The
        only variable to be intialized are the start and stopping wavelength bands.

        :param strtWvl:(float) (Default: 1.02)
        This parameter identifies in micrometer smallest wavelength band extracted by this class. (The first band is the
        wavelength closest to this value and maybe slightly lower or higher)

        :param stopWvl: (float) (Default: 2.6)
        This parameter identifies in micrometer largest wavelength band extracted by this class. (The final band is the
        wavelength closest to this value and maybe slightly lower or higher)
        """

        'Check the input parameters'
        assert isinstance(strtWvl, (int, float)) and (strtWvl < 2.602120), \
            "The starting wavelength must be a number less than 2.6 micrometer"
        assert isinstance(stopWvl, (int, float)) and (stopWvl > strtWvl), \
            "The stopping wavelength must be a number greater than the starting wavelength"

        'Set the starting wavelength'
        if strtWvl <= 1.021:
            print("The start wavelength is smaller than the limit for NIR Data resetting to 1.021 micrometers\n")
            self.__strtWvl = 1.021
        else:
            self.__strtWvl = strtWvl

        'Set the stopping wavelength'
        if strtWvl >= 2.602120:
            print("The stoping wavelength is Larger than the limit for NIR Data resetting to 2.602120 micrometers\n")
            self.__stopWvl = 2.602120
        else:
            self.__stopWvl = stopWvl

    def extract_nonFlatSpectra(self, img_name, output_mode='pandas'):
        """
        This function is used to extract spectra which do not appear completely flat from the denoised CRISM images.

        :param img_name: [string]
        This parameter is the address of the image which we are trying to process.

        :param output_mode: [string in ['pandas', 'numpy']] (Default: 'pandas')
        The output_mode controls whether the output data is a pandas dataframe or a numpy array

        :return: a pandas dataframe with the pixel positions the original spectra and its continuum
        """

        assert os.path.isfile(img_name), "The image provided does not exist"
        assert output_mode in ['pandas', 'numpy'], "Pick a valid output mode"

        'Read in the image'
        hsi_img, hsi_header = spectral_utilities().open_hsiImage(img_name, strt_wvl=self.__strtWvl,
                                                                 stop_wvl=self.__stopWvl)
        wvl = np.asarray(hsi_header["wavelength"], dtype=np.float32)
        wvl_names = ['{:1.4f}  microns'.format(item) for item in wvl]

        'Get the frame information'
        frm_info = spectral_utilities().extract_crismFrameInfo(hsi_img)

        'Estimate the continuum'
        hsi_cont = spectral_utilities().crism_contrem_mdl_nb(img_name, strt_wvl=self.__strtWvl,
                                                                 stop_wvl=self.__stopWvl)

        'Perform the continuum removal'
        hsi_crem = hsi_img / hsi_cont

        'Perform spatial smoothing'
        for ii in range(hsi_crem.shape[2]):
            t1 = np.squeeze(hsi_crem[frm_info['strtRow']:frm_info['stopRow'],
                            frm_info['strtCol']:frm_info['stopCol'], ii])
            t1 = ndimage.uniform_filter(t1, size=5)
            hsi_crem[frm_info['strtRow']:frm_info['stopRow'],
            frm_info['strtCol']:frm_info['stopCol'], ii] = t1

        'Estimate the mask from the continuum removed image'
        hsi_mask = crism_processing().crism_contrem_mask(hsi_crem, flat_thresh=0.02)

        'Find the pixels with significant absorptions'
        idx_sigabs = np.where(hsi_mask == 1)

        'Get the data'
        spectra_sigabs = hsi_img[idx_sigabs[0], idx_sigabs[1], :]
        cospectra_sigabs = hsi_cont[idx_sigabs[0], idx_sigabs[1], :]

        'Get the data location'
        if output_mode == 'pandas':
            img_info = np.vstack((np.asarray([os.path.basename(img_name)] * len(idx_sigabs[0])), idx_sigabs[0],
                                  idx_sigabs[1]))
            df_imginfo =  pd.DataFrame(img_info.T, columns=["Image Name", "Row", "Column"])

            df_spectral_data = pd.DataFrame(spectra_sigabs, columns=wvl_names)
            df_cospectral_data = pd.DataFrame(cospectra_sigabs, columns=wvl_names)

            return df_imginfo, df_spectral_data, df_cospectral_data
        else:
            img_info = np.vstack((np.asarray([os.path.basename(img_name)] * len(idx_sigabs[0])), idx_sigabs[0],
                                  idx_sigabs[1]))

            return img_info, spectra_sigabs, cospectra_sigabs


if __name__ == "__main__":
    'Create an object to extact non-flat spectra from CRISM images'
    obj1 = extract_samples(strtWvl=1.0275)

    'Test the sample extraction for a single CRISM Image'
    img_name = os.path.join('/Volume1/data/CRISM/atmCorr_stable/mica_images',
               'FRT0000A425', 'FRT0000A425_07_IF166L_TRRB_sabcondpub_v1_nr_ds.img')
    a, b, c = obj1.extract_nonFlatSpectra(img_name)

