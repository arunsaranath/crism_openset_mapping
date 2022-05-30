# -*- coding: utf-8 -*-

"""
FileName:               crism_mapping
Author Name:            Arun M Saranathan
Description:            This code file is used to analyze the various spectra identified as end-members by the automated
                        processing and tries to combine the end-members into one large library with the endmembers. I
                        will attempt to create groups from these end-members so an expert can label these endmembers
                        quickly.

Dependancies:           numpy, os, tqdm, spectral, hsiUtilities, spectral_utilities

Date Created:           02nd August 2021
Last Modified:          02nd August 2021
"""

'Import python core packages'
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
from spectral.io.envi import SpectralLibrary as sli
from sklearn.metrics import pairwise_distances

from hsiUtilities.crism_ganProc_utils import crism_ganProc_utils
from crism_processing.crism_processing import crism_processing
from spectral_utilities.spectral_utilities import spectral_utilities

class analyze_identifiedEM(object):
    """
    This class contains the set of functions compare spectra identified from various sources and groups/combines them
    and can be used to create final maps for the various images
    """
    def __init__(self, basesli_name, grp_label=None, feat_mode='none', feat_ext=None,
                 cont_rem=True, scl_lvl=None, strt_wvl=None, stop_wvl=None):
        """
        This constructor is used to create an object that contains the base set of endmembers and keeps track of the
        end-members that are being found/added.

        :param base_sliloc: [numpy.ndarray: nEM x nBands]
        This is a numpy matrix that contains the base end-members with which we are starting with.

        :param wavelength: [numpy.ndarray: nBands] (Default: None)
        The wavelength associated with the bands for the specific endmembers in question. If none is given the object
        creates on which assigns consecutive postive integers to each band in the image.

        :param grp_label: [numpy.ndarray: nGroups] (Default: None)
        Groups any novel spectra into groups based on similarity to speed up expert analysis. If None is the constructor
        assigns each endmember in the base library to a seperate group.

        :param feat_mode [string in ('none', 'param')] (Default: 'none')
        This decides whether the classification is directly performed on the data or if some param extraction is
        performed

        :param feat_ext: [Keras Model] (Default: None)
        This is a pretrained feature extactor which accepts the imput with the same shape as the input shape.

        :param cont_rem: (Boolean) [Default: True]
        This flag decides whether continuum removal needs to be performed for the spectra in the new library.

        :param scl_lvl: (0 <= np.float <= 1) [Default: 0.2]
        The level to which the spectra in the library are to be scaled.

        :param strt_wvl [float] (Default: None)
        The wavelength from which the image is being read. If None is given starts from the first band.

        :param stop_wvl [float] (Default: None)
        The wavelength upto which the image is being read. If None is given goes till band.
        """

        assert os.path.isfile(basesli_name), "The Spectral library provided does not exist"
        if grp_label is not None:
            assert isinstance(grp_label, np.ndarray) and (len(given_spectra.shape) == 1) and \
                   (grp_label.max() <= given_spectra.shape[0] - 1) and (grp_label.min() >= 0)\
                , "Group labels must be positive integers, with number of groups less than total number of spectra in" \
                  "the given endmembers"

        assert feat_mode in ['none', 'param'], "Unknown classification modes have been designed"
        if feat_ext is not None: assert isinstance(feat_ext, tf.keras.Model), "The representation model must be a " \
                                                                              "keras model"
        assert isinstance(cont_rem, bool), "The cont_rem variable must be Boolean"
        if scl_lvl is not None: assert (0 <= scl_lvl <= 1), "The scale level must also be between 0 & 1"

        self.scl_lvl = scl_lvl
        self.cont_rem= cont_rem
        self.feat_mode = feat_mode
        self.strt_wvl = strt_wvl
        self.stop_wvl = stop_wvl
        'Check feature extractor exists'
        if feat_ext is None:
            print('Since no feature extractor is provided!!! CHANGING TO NO FEAT MODE!!')
            self.feat_mode = 'none'
        else:
            self.feat_ext = feat_ext

        'Initialize variables of interest'
        self.given_spectra, hdr = spectral_utilities().prepare_slifiles(basesli_name, strt_wvl=self.strt_wvl,
                                                                      stop_wvl=self.stop_wvl)

        'If not given wavelength assign an integer to each band'
        if 'wavelength' in hdr:
            self.wavelength = np.asarray(hdr['wavelength'], dtype=np.float32)
        else:
            self.wavelength = np.arange(given_spectra.shape[1])

        'If not given names/labels for the spectra, assign generic names'
        if 'spectra names' in hdr:
            self.spectra_names = hdr['spectra names']
        else:
            self.spectra_names = [f"Spectrum_{ii}" for ii in self.given_spectra.shape[0]]

        'Extract the features for the exemplars'
        if self.feat_mode == 'param':
            if self.cont_rem:
                'Estimate the continuum for the spectra'
                given_cospectra = crism_processing().matrix_contrem_nb(self.given_spectra, self.wavelength)
                'Estimate the continuum removed spectra'
                self.given_crspectra = self.given_spectra / given_cospectra

            'If required also scale the continuum removed spectra  prior to feature extraction'
            if self.scl_lvl is not None:
                self.given_crspectra = spectral_utilities().scale_spectra_cr(self.given_crspectra, scale_lvl=self.scl_lvl)

            self.given_feat = self.feat_ext.predict(np.expand_dims(self.given_crspectra, axis=2))
        else:
            self.given_feat = self.given_spectra

        'If not given group labels assign each endmember to a different group'
        if 'grp_label' in hdr:
            self.grp_label = np.asarray(hdr['grp_label'], dtype=np.int)
        else:
            self.grp_label = np.arange(self.given_spectra.shape[0])


    def analyze_imageoutliers(self, sli_name, sig_thresh=0.95, grp_level=0.80):
        """
        This function is used to compare the outlier spectra in some spectral library are similar/different from the
        spectra in the base library. The function rejects spectra which are too close to existing members in the base
        library. Spectra which are of intermediate similarity are added to the base library but are assigned to the same
        group. Spectra which are very different are added to the base library and also assigned to a new group.

        :param sli_name: [string]
        The name of the spectral library which contains the spectra which we want to compare the base library

        :param sig_thresh: [0 <= np.float <= 1] (Default: 0.95)
        The (cosine) similarity level beyond which the model considers a new spectrum as very similar to spectra in the
        base library.

        :param grp_level: [0 <= np.float <= 1] (Default: 0.85)
        The (cosine) similarity level above which a model is considered to be in the same group as an existing spectrum.

        :return:
        """

        assert os.path.isfile(sli_name), "The spectral library specified does not exist"
        assert (0 <= sig_thresh <= 1), "The sig_thresh value must be in [0, 1]"
        assert (0 <= grp_level <= 1) and (grp_level < sig_thresh), "The grp_level must be in [0, 1] and grp_level must " \
                                                                   "be less than sig_thresh"

        'Get the naming key from the sli file'
        name_key = os.path.basename(sli_name)[:11]

        'First open the spectral library with appropriate continuum removal and scaling'
        exemplar_spectra, hdr = spectral_utilities().prepare_slifiles(sli_name, strt_wvl=self.strt_wvl,
                                                                      stop_wvl=self.stop_wvl)
        exemplar_spectra = np.asarray(exemplar_spectra, dtype=np.float32)

        'If using param mode- get the features of the exemplars'
        if self.feat_mode == 'param':
            if self.cont_rem:
                'Estimate the continuum for the spectra'
                exemplar_cospectra = crism_processing().matrix_contrem_nb(exemplar_spectra, self.wavelength)
                'Estimate the continuum removed spectra'
                exemplar_crspectra = exemplar_spectra / exemplar_cospectra

            'If required also scale the continuum removed spectra prior to feature extraction'
            if self.scl_lvl is not None:
                exemplar_crspectra = spectral_utilities().scale_spectra_cr(exemplar_crspectra, scale_lvl=self.scl_lvl)

            exemplar_feat = self.feat_ext.predict(np.expand_dims(exemplar_crspectra, axis=2))
        else:
            exemplar_feat = exemplar_crspectra

        'Check the spectra one at a time'
        for ii in range(exemplar_feat.shape[0]):
            'Get the distance of the exemplar'
            dist = np.squeeze(1 - pairwise_distances(exemplar_feat[ii, :].reshape(1, -1),
                                                     self.given_feat, metric='cosine'))
            'If there are no similar exemplars add this exemplar to the list'
            if np.max(dist) >= sig_thresh:
                print('This exemplar is very similar to library endmember at {:d}'.format(np.argmax(dist)))
            elif (grp_level <= np.max(dist) <= sig_thresh):
                'Append the spectrum to existing matrix with the endmembers with group of closest endmember'
                self.given_spectra = np.vstack((self.given_spectra, exemplar_spectra[ii, :].reshape(1, -1)))
                self.given_crspectra = np.vstack((self.given_crspectra, exemplar_crspectra[ii, :].reshape(1, -1)))
                self.given_feat = np.vstack((self.given_feat, exemplar_feat[ii, :].reshape(1, -1)))
                self.grp_label = np.hstack((self.grp_label, self.grp_label[np.argmax(dist)]))
                self.spectra_names = self.spectra_names + [(f"{name_key}_{ii}")]
            else:
                'Append the spectrum to the existing matrix as member of the new group'
                self.given_spectra = np.vstack((self.given_spectra, exemplar_spectra[ii, :].reshape(1, -1)))
                self.given_crspectra = np.vstack((self.given_crspectra, exemplar_crspectra[ii, :].reshape(1, -1)))
                self.given_feat = np.vstack((self.given_feat, exemplar_feat[ii, :].reshape(1, -1)))
                self.grp_label = np.hstack((self.grp_label, np.max(self.grp_label) + 1))
                self.spectra_names = self.spectra_names + [(f"{name_key}_{ii}")]

        return self

    def analyze_folder(self, folder_loc, sli_string=".sli", sig_thresh=0.95, grp_level=0.85,
                       cont_rem=True, strt_wvl=None, stop_wvl=None, save_flag=True, save_name="processed.sli"):
        """
        This function can be used to iterate over all the files inside the folder location provided and analyzes the
        end-member spectra contained in SLI which ends with a specific sequence

        :param folder_loc: (string)
        The string which is the address of a physical location which contains a bunch of images of the type we want to
        process.

        :param sli_string (string) (Default:'.sli')
        The sub-string which the SLI contain, which the function will analyze.

        :param sig_thresh: [0 <= np.float <= 1] (Default: 0.95)
        The (cosine) similarity level beyond which the model considers a new spectrum as very similar to spectra in the
        base library.

        :param grp_level: [0 <= np.float <= 1] (Default: 0.85)
        The (cosine) similarity level above which a model is considered to be in the same group as an existing spectrum.

        :param cont_rem: (Boolean) [Default: True]
        This flag decides whether continuum removal needs to be performed for the spectra in the new library.

        :param save_flag: (Boolean) [Default: True]

        :return:
        """

        assert os.path.isdir(folder_loc), "The specified folder does not exist"
        assert isinstance(sli_string, str), "The sli_string must be a string variable"
        assert isinstance(save_flag, bool), "The save_flag must be a boolean value"
        assert isinstance(save_name, str), 'The name of the sli to be saved must be a string'

        'Iterate over the files in the specified folder'
        for r, d, f in tqdm(os.walk(folder_loc)):
            for file in f:
                if file.find(sli_string) != -1:
                    'The SLI address is'
                    sli_address = os.path.join(r, file)

                    self.analyze_imageoutliers(sli_address, sig_thresh=sig_thresh, grp_level=grp_level)

        if save_flag:
            sli_hdr = {'wavelength': np.asarray(self.wavelength, dtype=np.float32),
                       'lines': self.given_spectra.shape[0], 'samples': self.given_spectra.shape[1], 'bands': 1,
                       'spectra names': self.spectra_names, 'grp_label':self.grp_label}
            'Create an SLI object'
            processed_sli = sli(self.given_spectra, sli_hdr, [])

            'If the directory mentioned in save does not exist make that directory'
            if not os.path.isdir(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))

            'Save the SLI'
            add_exem_sli = sli(self.given_spectra, sli_hdr, [])
            add_exem_sli.save(save_name)


        return self


if __name__ == "__main__":
    'Open and get the exemplar spectra'
    #sli_name = os.path.join('/Volume2/arunFiles/CRISM_minMapping_Ident/python_codeFiles',
    #                        'mica_processing/data_products/exemplars_r1.sli')
    sli_name = os.path.join(os.getcwd(), 'data_files/add_exemp_oyama_p2.sli')

    #exemplar_crspectra, hdr = spectral_utilities().prepare_slifiles(sli_name, strt_wvl=1.0275,
    #                                                                cont_rem=True, scl_lvl=0.2)
    'Create the feature extractor of interest'
    dis_rep = crism_ganProc_utils().create_rep_model()

    'Get the wavelengths associated with these spectra'
    #wvl = np.asarray(hdr['wavelength'], dtype=np.float32)

    'Create an object of the class to analyze the basic endmembers'
    obj1 = analyze_identifiedEM(sli_name, feat_mode='param', feat_ext=dis_rep, scl_lvl=0.2, strt_wvl=1.0275,
                                stop_wvl=2.6)

    'Get the endmembers from a specific image'
    """img_sli = os.path.join('/Volume1/data/CRISM/atmCorr_stable/mica_images/FRT00003E12',
                           'FRT00003E12_07_IF166L_TRRB_sabcondpub_v1_nr_ds_doc_class_v2_extracted_spectra.sli')
    obj1.analyze_imageoutliers(img_sli)"""

    img_folder = '/Volume1/data/CRISM/atmCorr_stable/oyama_crater/se_outsideChannel'
    sli_string = '_doc_class_v2_extracted_spectra.sli'
    save_name = os.path.join(os.getcwd(), 'data_files/add_exemp_oyama_p2')
    obj1.analyze_folder(img_folder, sli_string=sli_string, cont_rem=True, strt_wvl=1.0275, save_flag=True,
                        save_name=save_name)

    print('finished')
