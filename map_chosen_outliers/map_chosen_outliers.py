# -*- coding: utf-8 -*-

"""
FileName:               map_chosen_outliers
Author Name:            Arun M Saranathan
Description:            This code file is used to map the newly identfied outlier spectra by using a simple cosine
                        metric in the representation space

Dependacies:            numpy, spectral, keras (tensorflow), hsiU
"""
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics.pairwise import cosine_similarity as cosineDist
from spectral.io import envi
from tqdm import tqdm


from spectral_utilities.spectral_utilities import spectral_utilities
from hsiUtilities.crism_ganProc_utils import crism_ganProc_utils
from crism_processing.crism_processing import crism_processing

class map_chosen_outliers(object):
    def __init__(self, disrep_model, base_sliname, outlier_sliname, input_shape=None, scl_lvl=None):
        """
        This class contains the tools needed to map the presence of some additional outliers in some CRISM images. The
        class uses a feature extractor to get the features of both the newly created combined exemplars list and then
        maps the chosen images in a specified folder.

        :param disrep_model: (Keras Model) (Default: None)
        This is a pretrained feature extactor which accepts the imput with the same shape as the input shape.

        :param base_sliname: (str) (Default: None)
        The base set of exemplars which were mapped by the classification model.

        :param outlier_sliname: (str) (Default: None)
        The additional set of exemplars that were identified by the open-set classification model.

        :param input_shape: (Tuple: int) (Default: (240, 1,))
        This parameter is an integer list describing the shape of the input dataset. Default shape of the spectral
        data is set to 240 X 1, which is the size of NIR spectral data from the CRISM image database in the spectral
        range 1.0 to 2.6 microns.
        """

        assert isinstance(disrep_model, tf.keras.Model), "The representation model must be a keras model"
        self.feat_ext = disrep_model

        if input_shape is None:
            input_shape = (240, 1,)
        assert isinstance(input_shape, tuple) and all(isinstance(v, int) for v in input_shape), "This variable must" \
                                                                                                "be a tuple with " \
                                                                                                "integers"
        self.input_shape= input_shape
        if scl_lvl is not None:
            assert (0 <= scl_lvl <= 1), "The scaling level must be between 0 & 1"

        self.scl_lvl = scl_lvl

        'Read in the spectral library with the both the base and outlier exemplars'
        assert os.path.isfile(base_sliname), "The <base_sliname> must be the address string of the spectral library with the " \
                                         "base exemplars"

        assert os.path.isfile(outlier_sliname), "The <outlier_sliname> must be the address string of the spectral library " \
                                            "with the newly identified oulier-type exemplars"
        self.comb_sli, self.comb_hdr = spectral_utilities().combine_sli(base_sliname, outlier_sliname, save_flag=False,
                                                                        cont_rem=True, scl_lvl=0.2, strt_wvl=1.0275,
                                                                        stop_wvl=2.6)

        'Get the feature space representation of the exemplars'
        self.comb_feat = self.feat_ext.predict(np.expand_dims(self.comb_sli, axis=2))

    def crism_gan_mapping(self, img_name, filter_size=None, save_flag=False, strt_wvl=None,
                          stop_wvl=None, rem_noisy_pixel=True):
        """
        This function gets the spectral data from the CRISM image and generates maps based on the similarity in feature
        space made up of the representations learned from the GAN model.

        :param img_name: [string]
        The address of a crism image.

        :param filter_size: [int] (Default: None)
        The size of the smoothing filter applied to the image if needed. If filter_size is None then no  filter is
        applied to the images.

        :param save_flag: [Boolean] (Default: False)
        This flag decides whether the image is saved to the disk or not

        :param strt_wvl [float] (Default: None)
        The wavelength from which the image is being read. If None is given starts from the first band.

        :param stop_wvl [float] (Default: None)
        The wavelength upto which the image is being read. If None is given goes till band.

        :param rem_noisy_pixel: [bool] (Default: True)
        The noisiest 5% of pixels are excluded from the training

        :return
        """
        assert os.path.isfile(img_name), "The specified image does not exist"
        if filter_size is not None: assert isinstance(filter_size, int) and (filter_size > 0), \
            "The filter kernel size must be an integer"
        assert isinstance(save_flag, bool), "The save_flag must be either True or False"
        if strt_wvl is not None: assert isinstance(strt_wvl, (int, float)) and (strt_wvl < 2.602120), \
            "The starting wavelength must be a number less than 2.6 micrometer"
        if stop_wvl is not None: assert isinstance(stop_wvl, (int, float)) and (stop_wvl > strt_wvl), \
            "The stopping wavelength must be a number greater than the starting wavelength"
        assert isinstance(rem_noisy_pixel, bool), "The rem_noisy_pixel parameter must be a Boolean value"

        print('Processing the image {}\n'.format(os.path.basename(img_name)))

        'Read in the image'
        hsi_img, hsi_header = spectral_utilities().open_hsiImage(img_name, strt_wvl=strt_wvl, stop_wvl=stop_wvl)
        wvl = np.asarray(hsi_header["wavelength"], dtype=np.float32)

        'Get the frame info'
        frm_info = spectral_utilities().extract_crismFrameInfo(hsi_img)

        'Read in the model image'
        mdl_name = img_name.replace('_nr_ds.img', '_mdl_ds.img')
        mdl_img, mdl_hdr = spectral_utilities().open_hsiImage(mdl_name, strt_wvl=strt_wvl, stop_wvl=stop_wvl)



        'Estimate the continuum'
        hsi_cont = spectral_utilities().crism_contrem_mdl_nb(img_name, strt_wvl=strt_wvl, stop_wvl=stop_wvl)

        'Perform the continuum removal'
        hsi_crem = hsi_img / hsi_cont
        mdl_crem = mdl_img / hsi_cont

        'Smooth the continuum removed'
        hsi_crem = spectral_utilities().filter_hsi_bands(hsi_crem, filter_size=5)
        mdl_crem = spectral_utilities().filter_hsi_bands(mdl_crem, filter_size=5)

        'Estimate the mask from the continuum removed image'
        hsi_mask = crism_processing().crism_contrem_mask(hsi_crem, flat_thresh=0.01)

        if rem_noisy_pixel:
            'Estimate the residual image'
            res_mask = np.sum(np.abs(hsi_crem - mdl_crem), axis=2)

            'Find the threshold corresponding to 2 standard deviations above the mean'
            thresh = np.nanmean(res_mask) + 2 * np.nanstd(res_mask)
            'Apply thresh to mask'
            res_mask[res_mask > thresh] = 0
            res_mask[res_mask != 0] = 1

            'Update mask using the residual mask'
            hsi_mask = hsi_mask * res_mask

        'Find the pixels with significant absorptions'
        idx_sigabs = np.where(hsi_mask == 1)

        'Create a variable to hold the best guesses of the model'
        best_guess = np.zeros((hsi_crem.shape[0], hsi_crem.shape[1], self.comb_sli.shape[0]))

        if idx_sigabs[0].size != 0:
            'Get continuum removed spectra with significant absorptions'
            crspectra = hsi_crem[idx_sigabs[0], idx_sigabs[1], :]

            'Scale spectra if needed'
            # scl_lvl = 0.2
            if self.scl_lvl is not None:
                crspectra = spectral_utilities().scale_spectra_cr(crspectra, scale_lvl=self.scl_lvl)

            'Perform the open-set predictions for the significant spectra'
            spectra_pred = self.feat_ext.predict(np.expand_dims(crspectra, axis=2))

            'Find the cosine distance between the exemplars and the data'
            dist = np.squeeze(cosineDist(self.comb_feat, spectra_pred))


            'Zero out all except the highest value or best guess'
            dist = dist * (dist >= np.sort(dist, axis=0)[-1, :]).astype(float)

            'Threshold out the low values'
            dist[dist <= 0.707] = 0

            best_guess[idx_sigabs[0], idx_sigabs[1], :] = dist.T

            'Save the image and associated quantities if needed'
            if save_flag:
                'Change the save folder if needed'
                out_name = img_name.replace('.img', '_gan_maps_best_guess.hdr')
                out_name = out_name.replace('.IMG', '_gan_maps_best_guess.hdr')


                'Create a header for the doc classifier'
                header = {"Description": "Maps of the various outlier clusters identified automatically",
                          "key": "+ve - value identification, -1 - outlier"}
                header["lines"] = hsi_header["lines"]
                header["samples"] = hsi_header["samples"]
                header["bands"] = best_guess.shape[2]
                if self.comb_hdr["spectra names"] is not None:
                    header["spectra names"] = self.comb_hdr["spectra names"]

                'Save the image'
                envi.save_image(out_name, best_guess, dtype=np.float32, force=True,
                                interleave='bil', metadata=header)
                
        return best_guess

    def create_composite_detection_maps(self, best_guess_name, col_mat=None, ident_level=0.95, guess_level=0.85,
                                        save_flag=True):
        """
        This function is used to create a composite detection map from a given best guess image. Pixels which are ident-
        ified with high confidence are shown in bold colors while low-confidence detections are shown in subdued colors.

        :param best_guess_name: [string
        The address of the best-guess image from which we are creating a detection maps.

        :param col_mat: [ndarray: nBands X 3](Default: None)
        The color key which identifies the color assigned to a specific mineral class.

        :param ident_level: [0.707 <= float <= 1] (Default: 0.95)
        The similarity level above which a pixel is considered a high confidence detection

        :param guess_level: [0.707 <= float <= 1] (Default: 0.85)
        The similarity level above which a pixel is considered a detection

        :param save_flag: [Boolean] (Default: False)
        This flag decides whether the image is saved to the disk or not
        :return:
        """

        assert os.path.isfile(best_guess_name), "The sepecified best-guess image does not exist"
        'get the best guess image'
        bg_img, bg_hdr = spectral_utilities().open_hsiImage(best_guess_name)

        assert isinstance(col_mat, np.ndarray) and (len(col_mat.shape) == 2), "The <col_mat> must be a 2D matrix"
        assert col_mat.shape[0] == bg_img.shape[2], "<col_mat> must have a color assigned to each mineral class"
        assert col_mat.shape[1] == 3, "Assign an RGB color to each class"
        assert (0.707 <= ident_level <= 1.), "The <ident_level> must be in the range [0.707, 1]"
        assert (0.707 <= guess_level <= ident_level), "The <guess_level> must be in the range [0.707, <ident_level>]"


        'Create a detection maps'
        det_maps = np.zeros((bg_img.shape[0], bg_img.shape[1], 3))

        'Check each pixel individually'
        for ii in tqdm(range(bg_img.shape[0])):
            for jj in range(bg_img.shape[1]):
                'Check if the best-guess is greater lowest acceptable threshold'
                if np.max(np.squeeze(bg_img[ii, jj, :])) > guess_level:
                    'Check if best guess is higher that high confidence threshold'
                    if np.max(np.squeeze(bg_img[ii, jj, :])) > ident_level:
                        'Put bold color in the maps'
                        det_maps[ii, jj, :] = col_mat[np.argmax(np.squeeze(bg_img[ii, jj, :])), :]
                    else:
                        'Put subdued color in the image'
                        det_maps[ii, jj, :] = 0.6 * col_mat[np.argmax(np.squeeze(bg_img[ii, jj, :])), :]

        if save_flag:
            out_name = best_guess_name.replace("_compBestGuess_updated.img", "_detectMaps_updated.hdr")
            'Create a header for the doc classifier'
            header = {"Description": "Maps with specific RGB combination assigned to specific minerals, high confidence"
                                     "detection are in bold colrs while, lower confidence detection are in subdued"
                                     "colors."}
            header["lines"] = det_maps.shape[0]
            header["samples"] = det_maps.shape[1]
            header["bands"] = det_maps.shape[2]

            'Save the image'
            envi.save_image(out_name, det_maps, dtype=np.float32, force=True, interleave='bil', metadata=header)

        return det_maps



if __name__ == "__main__":

    'Set the filename of the the base set of outliers'
    base_sliname = "/Volume2/arunFiles/CRISM_minMapping_Ident/python_codeFiles/mica_processing/data_products/" \
                   "exemplars_r1.sli"
    outlier_sliname = "/Volume1/data/CRISM/atmCorr_stable/jezero/FRT00009E72/" \
                      "FRT00009E72_07_IF168L_TRRB_sabcondpub_v1_nr_ds_doc_class_v2_extracted_spectra.sli"

    'Create the feature extractor of interest'
    dis_rep = crism_ganProc_utils().create_rep_model()

    "Create the object for the GAN based mapping"
    obj1 = map_chosen_outliers(disrep_model=dis_rep, base_sliname=base_sliname, outlier_sliname=outlier_sliname,
                               scl_lvl=0.2)

    """
    'Set the image name'
    img_name  = "/Volume1/data/CRISM/atmCorr_stable/jezero/FRT00009E72/" \
                "FRT00009E72_07_IF168L_TRRB_sabcondpub_v1_nr_ds.img"
    
    _ = obj1.crism_gan_mapping(img_name, filter_size=5, save_flag=True, strt_wvl=1.0275, rem_noisy_pixel=True)
    """

    srcFolder = "/Volume2/data/CRISM/AMS/v3_IusChasma/"

    col_mat = np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
                          , [0, 0, 0], [0, 0, 0], [0, 0, 0], [65, 105, 205], [0, 255, 255], [0, 255, 255], [0, 0, 0]
                          , [0, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [192, 192, 192], [0, 0, 0]
                          , [218, 165, 32], [218, 165, 32], [127, 0, 255], [255, 105, 180], [255, 0, 0]],
                         dtype=np.float)

    for r, d, f in os.walk(srcFolder):
        for file in f:
            if file.find('_compBestGuess_updated.img') != -1:
                _=  obj1.create_composite_detection_maps(os.path.join(r, file), col_mat=col_mat)


