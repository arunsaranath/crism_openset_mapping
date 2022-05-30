# -*- coding: utf-8 -*-

"""
FileName:               basic_utilities
Author Name:            Arun M Saranathan
Description:            This code file contains the basic utilities needed to interact with CRISM type hyperspectral
                        data.
Dependancies:           Python Spectral package

Date Created:           19th June 2021
Last Modified:          19th June 2021
"""

import os
import numpy as np
from spectral.io import envi
from spectral.io.envi import SpectralLibrary as sli
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

from crism_processing.crism_processing import crism_processing

class spectral_utilities:
    def open_hsiImage(self, img_name, hdr_name=None, strt_wvl=None, stop_wvl=None):
        """
        This function is used to open an hyperspectral image. If a seperate header file address is not provided. The
        function assumes that there exists a header file associated with the hyperspectral in the same folder as the image.
        The header file can have an extension of either '.hdr', '.img.hdr', '.HDR' or '.IMG.HDR'. Any other extension will
        have to be explicitly provided. The function also allows you to open spectra in a region of interest

        :param img_name: (string)
        A string which provides the address of the image to be read.

        :param hdr_name: (string) (Default: None)
        A string which provides the address of the header file associated with the image to be read.

        :param strt_wvl: (float) (Default: None)
        The starting wavelength in the opened image

        :param stop_wvl: (float) (Default: None)
        The stopping wavelength in the opened image

        :return: image (3D:ndarray), header(dictionary)
        """
        'Check the input parameters'
        if strt_wvl is not None:
            assert isinstance(strt_wvl, (int, float)) and (strt_wvl < 2.602120), \
            "The starting wavelength must be a number less than 2.6 micrometer"
        if stop_wvl is not None:
            assert isinstance(stop_wvl, (int, float)) and (stop_wvl > strt_wvl), \
            "The stopping wavelength must be a number greater than the starting wavelength"

        assert os.path.isfile(img_name), "The image does not exist"
        assert (hdr_name is None) or os.path.isfile(hdr_name), "The provided header name does not exist"


        if hdr_name is None:
            'Get the header associated with this image'
            hdr_name = img_name.replace(".img", ".hdr")
            hdr_name = hdr_name.replace(".IMG", ".hdr")
            'Check if this file exists'
            if not os.path.isfile(hdr_name):
                hdr_name = img_name.replace(".img", ".img.hdr")
                hdr_name = hdr_name.replace(".IMG", ".img.hdr")
                if not os.path.isfile(hdr_name):
                    hdr_name = img_name.replace(".img", ".HDR")
                    hdr_name = hdr_name.replace(".IMG", ".HDR")
                    if not os.path.isfile(hdr_name):
                        hdr_name = img_name.replace(".IMG", "IMG.HDR")
                        hdr_name = hdr_name.replace(".img", "IMG.HDR")
                        if not os.path.isfile(hdr_name):
                            assert True, \
                                "The function could not find the associated header file. Please specify the HDR file"

        'Open/Read the Image'
        img = envi.open(hdr_name, img_name)
        cube = img.load()
        header = envi.read_envi_header(hdr_name)

        'First check if there is a wavelength field in the header'
        if "wavelength" in header:
            wvl = np.asarray(header["wavelength"], dtype=np.float64)
            'Find the band closest to the starting wavelength'
            if strt_wvl is not None:
                strt_band = np.argmin(np.abs(wvl - strt_wvl))
            else:
                strt_band = 0

            'Find the band closest to the stopping wavelength'
            if stop_wvl is not None:
                stop_band = np.argmin(np.abs(wvl - stop_wvl)) + 1
            else:
                stop_band = cube.shape[2]

        else:
            strt_band = 0
            stop_band = cube.shape[2]

        'Subset the data'
        cube = cube[:, :, strt_band:stop_band]
        'Update the header'
        header["bands"] = cube.shape[2]
        if "wavelength" in header:
            header["wavelength"] = wvl[strt_band:stop_band]

        return cube, header

    def extract_crismFrameInfo(self, img_cube):
        """
        This function returns the frame information of an image cube

        :param img_cube: [ndarray: nRows X nCols X nSamples]
        The image cube which we are analyzing

        :return:
        """

        'Perform Nan-Sum on the image cube'
        img_sum = np.nansum(img_cube, axis=2)

        'Find the non-zero values'
        nz_val = np.where(img_sum != 0)

        return {"strtRow":nz_val[0][0], "stopRow":nz_val[0][-1], "strtCol":nz_val[1][0], "stopCol":nz_val[1][-1]}

    def scale_spectra_cr(self, data, scale_lvl=0.02, zero_center=False):
        """
        This function scales continuum removed spectra such that the difference between the maximum and minimum values
        of each spectrum is equal to 'scale_lvl'

        :param data: [ndarray: nRows X nBands]
        A numpy matrix where the rows are individual spectra.

        :param scale_lvl: (float) (Default: 0.02)
        The difference between the maximum and minumum values

        :param zero_center: (bool) (default = False)
        Each row is shifted to be centered at 0

        :return: [ndarray: nRows X nBands]
        The scaled matrix
        """

        #assert (np.any(data[0, :] != 1)), "The image should be continuum removed"
        assert (0 <= scale_lvl <= 1), "The scale level must be a value between 0 & 1"
        assert isinstance(zero_center, bool),"The zero_center value must True or False"

        'If it is a 1D matrix make it a 2D matrix'
        if len(data.shape) == 1:
            data = data.reshape((1, -1))

        assert len(data.shape) == 2

        'First subtract 1 and set everything at 1 to 0'
        data_shft = data - 1
        'divide by the minimum in each row'
        data_min = data_shft.min(axis=1)

        data_scale = np.zeros(data.shape)

        'Scale each endmember and create plots to see what it looks like'
        for ii in range(data.shape[0]):
            temp = data_shft[ii, :] / data_min[ii]
            data_scale[ii, :] = temp * -1 * scale_lvl

        if zero_center:
            data_scale = data_scale - (1 - (scale_lvl/2))

        return (data_scale + 1)

    def prepare_slifiles(self, sli_filename, sli_hdrname=None, cont_rem=False, scl_lvl=None, strt_wvl=None,
                         stop_wvl=None):
        """
        This function is used to read in a spectral libraries. And preform some basic pre-processing on the spectra in
        these spectral libraries.

        :param sli_filename: (string)
        This is string which contains the address of the ENVI SLI file that is to be processed.

        :param sli_hdrName: (string) [Default: None]
        This is string which contains the address of the header associated with the SLI file that is to be processed.
        If none is provided it will be assumed the header has the name and same location as the image with  an
        extension of either '.sli.hdr' ,'.hdr', '.SLI.HDR' or 'HDR'.

        :param cont_rem: (Boolean) (Default: False)
        A boolean variable that indicates whether continuum removal needs to be performed on the spectra in the spectral
        library

        :param scl_lvl: (0 < float 1) (Default: None)
        A float variable that indicates the size of the largest absorption band in the spectrum. If no value is provided
        (default) no scaling is done. If it is provided the size of the largest absorption is equal to sclLvl. This
        parameter is in (0, 1]

        :return: (ndarray: nSamples X nBands)
        This variable contains the spectral data from the spectral library, each row is the spectrum of one the samples
        from the spectral library.
        """
        if scl_lvl is not None:
            assert (0. <= scl_lvl <=1), "The value of scale level must be between 0 & 1"
        assert isinstance(cont_rem, bool), "This variable must either be True or False"
        assert os.path.isfile(sli_filename), "This file does not exist!!"
        assert (sli_hdrname is None) or (os.path.isfile(sli_hdrname)), "This file does not exist!!"

        if sli_hdrname is None:
            'Get the header associated with this image'
            sli_hdrname = sli_filename.replace(".sli", ".hdr")
            sli_hdrname = sli_hdrname.replace(".SLI", ".hdr")
            'Check if this file exists'
            if not os.path.isfile(sli_hdrname):
                sli_hdrname = sli_filename.replace(".sli", ".sli.hdr")
                sli_hdrname = sli_hdrname.replace(".SLI", ".sli.hdr")
                if not os.path.isfile(sli_hdrname):
                    sli_hdrname = sli_filename.replace(".sli", ".HDR")
                    sli_hdrname = sli_hdrname.replace(".SLI", ".HDR")
                    if not os.path.isfile(sli_hdrname):
                        sli_hdrname = sli_filename.replace(".SLI", "SLI.HDR")
                        sli_hdrname = sli_hdrname.replace(".sli", "SLI.HDR")
                        if not os.path.isfile(sli_hdrname):
                            assert True, \
                                "The function could not find the associated header file. Please specify the HDR file"

        'Open the SLI file and extract the spectra'
        SLI = envi.open(sli_hdrname, sli_filename)
        spectra = SLI.spectra
        header = envi.read_envi_header(sli_hdrname)

        'First check if there is a wavelength field in the header'
        if "wavelength" in header:
            wvl = np.asarray(header["wavelength"], dtype=np.float32)
            'Find the band closest to the starting wavelength'
            if strt_wvl is not None:
                strt_band = np.argmin(np.abs(wvl - strt_wvl))
            else:
                strt_band = 0

            'Find the band closest to the stopping wavelength'
            if stop_wvl is not None:
                stop_band = np.argmin(np.abs(wvl - stop_wvl)) + 1
            else:
                stop_band = spectra.shape[1]

        'Perform the subsetting if needed'
        spectra = np.asarray(spectra[:, strt_band:stop_band], dtype=np.float32)
        wvl = wvl[strt_band:stop_band]
        header["wavelength"] = [str(wl) for wl in wvl]
        header["samples"] = str(len(wvl))

        'If necessary perform continuum removal'
        if cont_rem:
            'Estimate the continuum for the spectra'
            cospectra = crism_processing().matrix_contrem_nb(spectra, wvl=wvl)
            'Estimate the continuum removed spectra'
            crspectra = spectra / cospectra
        else:
            crspectra = spectra

        'Scale the spectra if needed'
        if scl_lvl is not None:
            crspectra = self.scale_spectra_cr(crspectra, scale_lvl=scl_lvl)

        return crspectra, header

    def combine_sli(self, sli1_name, sli2_name, save_flag=True, save_add= os.getcwd(), save_name='combined',
                    cont_rem=False, scl_lvl=None, strt_wvl=None, stop_wvl=None):
        """
        This function can be used to combine two different SLI (Assumes the header has the same name but ends with .hdr
        instead of .sli.). The function also assumes that so long as the two spectral libraries have the same number of
        bands that the wavelengths are the same, i.e. the function does not perform a seperate check to see if the two
        SLI files have the same wavelengths.

        :param sli1_Name: (string)
        Physical Address of the first SLI

        :param sli2_Name: (string)
        Physical Address of the second SLI

        :param save_flag: (boolean) (Default: True)
        Flag indicating whether the combined SLI is to be saved

        :param save_add: (string) (Default= cwd)
        The location where the combined SLI is to be saved.

        :parame save_name: (string)(Default= combined.sli)
        name of the combined sli
        :return:
        """

        assert os.path.isfile(sli1_name), "The first SLI does not exist"
        assert os.path.isfile(sli2_name), "The second SLI does not exist"
        assert isinstance(save_flag, bool), "The save_flag must be boolean"
        assert os.path.exists(save_add), "The saving location provided to the function does not exist"
        assert isinstance(save_name, str) , "The save_name parameter must a be a string"

        'Read in the two SLI'
        sli1, hdr1 = self.prepare_slifiles(sli1_name, cont_rem=cont_rem, scl_lvl=scl_lvl, strt_wvl=strt_wvl,
                                           stop_wvl=stop_wvl)
        sli2, hdr2 = self.prepare_slifiles(sli2_name, cont_rem=cont_rem, scl_lvl=scl_lvl, strt_wvl=strt_wvl,
                                           stop_wvl=stop_wvl)

        'Intialize and header for the Spectral library'
        sli_hdr = hdr1

        'Check if they have matching properties'
        if (hdr1['samples'] == hdr2['samples']):
            'Combine the two sets of spectra'
            sli_spectra = np.vstack((sli1, sli2))

            'Create the appropriate metadata'
            'Header details'
            sli_hdr['lines'] = sli_spectra.shape[0]
            sli_hdr['samples'] = sli_spectra.shape[1]
            sli_hdr['spectra names'] = hdr1['spectra names'] + hdr2['spectra names']

            if save_flag:
                '.sli with MICA numerators'
                micaNumSli = sli(sli_spectra, sli_hdr, [])
                micaNumSli.save(os.path.join(save_add, save_name))

        else:
            raise Exception('The two SLI have files contain spectra with differing number of bands')


        return sli_spectra, sli_hdr

    def crism_contrem_mdl_nb(self, img_name, model_img_name=None, filter_size=None, strt_wvl=None, stop_wvl=None):
        """
        In the case of Yuki's denoising model [1] for each CRISM image there is an approximate model which does not have
        any noise/distortions. In many cases, the continuum removal is better performed on this noiseless spectral data
        rather than the original noisy image. The function assumes that images denoised by Yuki's method will have the
        string token '_nr_ds' in the image name. If no model image name is provided then the function assumes that the
        model image replaces the string token '_nr_ds' with the string token '_mdl_ds'. The function will find the model
        file and performs continuum removal on the model image data

        [1] Itoh, Y. and Parente, M., "A new method for atmospheric correction and de-noising of CRISM hyperspectral
        data", Icarus 354 (2021): 114024

        :param img_name: [string]
        This is string which is the address of the CRISM denoised image.

        :param model_img_name: [string] (Default: None)
        This is string which is the address of the CRISM model image. If no parameter is given the function looks for
        the string '_nr_ds' in the image name with the string '_mdl_ds'.

        :param filter_size: [int] (Default: None)
        The size of the spectral smoothing filter that is to be applied if needed. If no value is given then spectral
        smoothing is not performed

        :param strt_wvl: [float] (Default: None)
        The wavelength from which the CRISM data is extracted

        :param stop_wvl: [float] (Default: None)
        The wavelength till which the CRISM data is extracted

        :return:
        """

        'Check input variables'
        assert os.path.isfile(img_name), "No file exists with the given image name"
        if model_img_name is None:
            assert ('_nr_ds' in img_name), "The file name does not follow default model structure"
            model_img_name = img_name.replace('_nr_ds', '_mdl_ds')
        assert os.path.isfile(model_img_name), "Cannot find a model image with default name"
        if filter_size is not None:
            assert isinstance(filter_size, int) and (filter_size > 0), "The filter size (if specified) must be a" \
                                                                       " positive integer"
        if strt_wvl is not None:
            assert isinstance(strt_wvl, float), "The starting wavelength must be a floating point number"

        if stop_wvl is not None:
            assert isinstance(stop_wvl, float), "The stopping wavelength must be a floating point number"

        'Read in the model image cube'
        model_img, model_header = self.open_hsiImage(model_img_name, strt_wvl=strt_wvl,
                                                                 stop_wvl=stop_wvl)
        wvl = np.asarray(model_header["wavelength"], dtype=np.float32)

        'Get the frame information'
        frm_info = spectral_utilities().extract_crismFrameInfo(model_img)

        'Estimate the continuum'
        return crism_processing().crism_contRem_nb(model_img, wvl, frm_info=frm_info)

    def crism_create_mdlimg(self, abimg_name, bgimg_name=None, out_flag= False, out_name =None):
        """
        This function can be used to create the modelled image from the absorption and background components.

        :param abimg_name: [str]
        String denoting the address of the image which contains the modeled absorption spectrum for a specific CRISM
        image.

        :param bgimg_name: [str] [Default: None]
        String denoting the address of the image which contains the modeled absorption spectrum for a specific CRISM
        image. If none is provided the function assumes that the background image replaces the string "_AB." with
        the string "_Bg"

        :param out_flag [bool] (Default: False)
        A boolean flag that decides whether the combined image is stored or not

        :param out_name [str] (Default: out.img)
        If <out_flag> is True, the combined image is stored under this name. If None is provided it is saved in the cwd
        as  "out.img"

        :return:
        """

        assert os.path.isfile(abimg_name), (f"No file located at {abimg_name}")
        if bgimg_name is None:
            bgimg_name = abimg_name.replace('_AB_ds.', '_Bg_ds.')
        assert os.path.isfile(bgimg_name), (f"Could not locate the background image at {bgimg_name}")
        assert isinstance(out_flag, bool), "The parameter <outflag> must be boolean"
        if out_flag:
            if out_name is None: out_name = os.path.join(os.getcwd(), 'out.img')
        assert os.path.isdir(os.path.dirname(out_name)), "The location to save the combined file does not exist!!"

        'Read in the image which contains the modeled absorption spectrum'
        ab_img, ab_hdr = self.open_hsiImage(abimg_name)

        'Read the image which contains the modeled background spectrum'
        bg_img, bg_hdr = self.open_hsiImage(bgimg_name)

        'Combine them together to create the modeled image name'
        mdl_img = ab_img * bg_img

        if out_flag:
            envi.save_image(out_name, mdl_img, dtype=np.float32, force=True, interleave='bil',
                            metadata=ab_hdr)

    def stack_plots(self, spectra, offset=0.001):
        """
        This function can be used to stack spectral plots for visualization.

        :param spectra: [ndarray: nSamples X nBands]
        This variable contains the spectra for visualization. Each row corresponds to one spectrum.

        :param offset: [float] (Default: 0.001)
        This variable controls the offset between the various spectra

        :return: spectra_stack [ndarray: nSamples X nBands]
        The spectra with the first being highest and so on
        """

        assert isinstance(spectra, np.ndarray) and (len(spectra.shape) == 2), "The <spectra> variable must be a 2D" \
                                                                              " numpy array"
        assert isinstance(offset, float) and (offset > 0.), "The <offset> variable must be a +ve float"

        'Get the number of spectra'
        nSamp = spectra.shape[0]

        'Create a variable to hold the stacked spectra'
        spectra_stack = np.zeros(spectra.shape)

        for ii in range(nSamp):
            'The spectrum currently being processed is '
            idx = nSamp - ii - 1

            if(ii == 0):
                spectra_stack[idx, :] = spectra[idx, :]
            else:
                spectrum = (spectra[idx, :] - (spectra[idx, :]).min()) + spectra_stack.max() + offset
                spectra_stack[idx, :] = spectrum

        return spectra_stack

    def filter_hsi_bands(self, img_cube, filter_size=None, filter_kernel=None):
        """
        This function can be used to apply various 2D filters to each individual band of a HSI (or a 3D numpy array).

        :param img_cube: [ndarray: nLines X nSamples X nBands]
        This the HSI image cube to be processed

        :param filter_size: [int] (Default: 5)
        The size/footprint of the filter. If none is provided the function will set this value to 5.

        :param filter_kernel: [ndarray: filter_size X filter_size] (Default: averaging filter)
        The actual filter characterstics. If none is provide the function will assume an averaging/smoothing filter.

        :return:
        """

        assert isinstance(img_cube, np.ndarray), "The <img_cube> must be a numpy array"
        assert len(img_cube.shape) == 3, "The <img_cube> must be a 3D array"
        if filter_size is not None:
            assert isinstance(filter_size, int) and (filter_size >0), "<filter_size> must be a positive integer"
        else:
            filter_size = 5

        if filter_kernel is not None:
            assert isinstance(filter_kernel, np.ndarray) and len(filter_kernel.shape) == 2, \
                "<filter_kernel> must be a 2D numpy array"
            assert (filter_kernel.shape[0] < img_cube.shape[0]) and (filter_kernel.shape[1] < img_cube.shape[1]),\
                f"<filter_kernel> dimensions must be smaller than the cube dimensions " \
                f"({img_cube.shape[0], img_cube.shape[1]})"
        else:
            filter_kernel = (1. / filter_size**2) * np.ones((filter_size, filter_size))

        'Get the frame information'
        frm_info = self.extract_crismFrameInfo(img_cube)

        for ii in range(img_cube.shape[2]):
            'Get a single band'
            t1 = np.squeeze(img_cube[frm_info["strtRow"]:frm_info["stopRow"],
                            frm_info["strtCol"]:frm_info["stopCol"], ii])

            'Apply the covolution kernel'
            t1 = ndimage.convolve(t1, filter_kernel, mode="nearest")

            'Replace the smoothed version in the cube'
            img_cube[frm_info["strtRow"]:frm_info["stopRow"], frm_info["strtCol"]:frm_info["stopCol"], ii] = t1

        return img_cube






if __name__ == "__main__":
    'Create a spectral utilities object'
    obj1 = spectral_utilities()

    'Set the name of the SLI to be processed'
    sli_name = "/Volume2/arunFiles/CRISM_minMapping_Ident/ident4_pres/CMS_detections/kiserite.sli"

    "Read in the SLI contents"
    given_spectra, hdr = obj1.prepare_slifiles(sli_name, strt_wvl=1.0275, stop_wvl=2.6, cont_rem=True, scl_lvl=0.2)
    "Extract meta data"
    wavelength = np.asarray(hdr['wavelength'], dtype=np.float32)
    spectra_names = hdr['spectra names']

    given_spectra_stack = obj1.stack_plots(given_spectra)

    print('finished')

    """
    --------------------------------------------------------------------------------------------------------------------
    Create model images
    
    base_loc = '/Volume1/data/CRISM/atmCorr_stable/gale'

    for r, d, f in os.walk(base_loc):
        for file in f:
            if file.find("_AB_ds.img") != -1:
                'The image address is'
                img_address = os.path.join(r, file)

                'Get the name for the output file'
                out_name= img_address.replace("_AB_ds.img", "_mdl_ds.hdr")

                'Call the function to create the modeled image'
                obj1.crism_create_mdlimg(img_address, out_flag=True, out_name=out_name)
    """


    """
    --------------------------------------------------------------------------------------------------------------------
    Combine Spectral Libraries
    """
    """
    sli_lib1 = os.path.join('/Volume1/data/CRISM/atmCorr_stable/mawrth_vallis/manual_analysis',
                            'exemplar_r1.sli')

    sli_lib2 = os.path.join('/Volume1/data/CRISM/atmCorr_stable/mawrth_vallis/manual_analysis',
                            'manually_selected_EM.sli')

    save_fldr = '/Volume1/data/CRISM/atmCorr_stable/mawrth_vallis/manual_analysis'

    obj1.combine_sli(sli_lib1, sli_lib2, save_flag=True, save_add=save_fldr, save_name='mawrth_vallis_exemplar')
    """


    """
    -------------------------------------------------------------------------------------------------------------------
    'Now get its DOC map'
    map_name = os.path.join('/Volume1/data/CRISM/atmCorr_stable/mica_images',
                            'FRT0000ABCB', 'FRT0000ABCB_07_IF166L_TRRB_sabcondpub_v1_nr_ds_doc_class_v2.img')
    map_cube, map_hdr = obj1.open_hsiImage(map_name)

    print(map_cube.shape)
    """

