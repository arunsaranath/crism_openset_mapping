# -*- coding: utf-8 -*-

"""
FileName:               crism_mapping
Author Name:            Arun M Saranathan
Description:            This code file is used to extract samples the CRISM Images and then classify them using the DOC
                        to seperate the minerals of the known classes from the novel/unknown samples. Finally, the
                        number of classes in the unknown samples is estimated by using the method described in [1]. Then
                        a simple k-means is used identify the means corresponding to the unknown classes. These class-
                        means are then compared to the known library endmembers to see how these novel spectra compare.

                        [1] Han, K., Vedaldi, A., and Zisserman, A., 2019, "Learning to discover novel visual
                        categories via deep transfer clustering." In Proceedings of the IEEE/CVF International
                        Conference on Computer Vision, (pp. 8401-8409).

Dependancies:           numba, spectral, doc_crism_classifier, spectral_utilities

Date Created:           28th June 2021
Last Modified:          02nd August 2021
"""

'Import python core packages'
import spectral.io.envi as envi
import numpy as np
import os
from tqdm import tqdm
from scipy import ndimage
from spectral.io import envi
from spectral.io.envi import SpectralLibrary as sli
from sklearn import cluster
from kneed import KneeLocator
from sklearn.metrics import pairwise_distances

'Import user defined packages'
from doc_crism_classifier import doc_crism_classifier
from spectral_utilities.spectral_utilities import spectral_utilities
from crism_processing.crism_processing import crism_processing
from hsiUtilities.crism_ganProc_utils import crism_ganProc_utils

class crism_mapping(object):
    def __init__(self, doc_model=None, base_folder=None, target_folder=None):
        """
        This class can be used to map various CRISM images. The object assumes that it a Deep Open Classifier which can
        be used classify each significant pixel as either a member of a known class or an unknown class. Also if needed
        the function save the results of the classification in the same folder as the image. A modification of this is
        in case a base and target folders, the base folder is replaced by the target folder.

        :param doc_model: [doc_crism_classifier] (Default: None)
        The open set classification model is an instance of the doc_crism_classifier. If None is provided the class by
        default loads the classifier with the name 'doc_ganRep_crismData-1'.

        :param base_folder: [string] (Default)
        """

        if doc_model is not None: assert isinstance(doc_model, doc_crism_classifier), "The provided model must be a " \
                                                                                   "member of the " \
                                                                                   "'doc_crism_classifier' class."
        if base_folder is not None: assert os.path.isdir(base_folder), "The base folder does not exist"
        if target_folder is not None:
            if not os.path.isdir(target_folder):
                print('Target Folder does not exist! Creating folders')
                os.makedirs(target_folder)

        'If none is provided load the default deep open classifier'
        if doc_model is None:
            'Create a doc object with some classifier'
            disrep_model = crism_ganProc_utils().create_rep_model()
            self.doc = doc_crism_classifier(int(10), disrep_model=disrep_model)

            'Load a pretrained model'
            self.doc.load_preTrained_model(model_loc=os.path.join('/Volume2/arunFiles/CRISM_minMapping_Ident',
                                                                  'python_codeFiles/doc_crism_classifier',
                                                                  'doc_ganRep_crismData-1/doc_model'))
        else:
            'Load the given model'
            self.doc = doc_model

        'Set the variable for the base and target folders'
        self.base_folder = base_folder
        self.target_folder = target_folder


    def crism_doc_mapping(self, img_name, scl_lvl=None, scale=1., filter_size=None, save_flag=False, strt_wvl=None,
                          stop_wvl=None, rem_noisy_pixel=True, save_anc_prdct=False):
        """
        This function gets the spectral data from the CRISM image and generates maps from the image.

        :param img_name: [string]
        The address of a crism image.

        :param scl_lvl: [0 <= float <= 1] (Default: None)
        The level to which a continuum removed spectrum is scaled. By default no scaling is performed.

        :param scale: [float] (Default: 1.0)
        How many standard deviations are considered before a sample is detected as outlier.

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

        :param save_anc_prdct: [bool] (Default: True)
        This flag decide whether the ancilliary products that is the novel outlier means and maps of those means are
        saved

        :return:
        """

        assert os.path.isfile(img_name), "The specified image does not exist"
        if scl_lvl is not None: assert (0 <= scl_lvl <= 1), "The scaling level must be between 0 & 1"
        if filter_size is not None: assert isinstance(filter_size, int) and (filter_size > 0), \
            "The filter kernel size must be an integer"
        assert isinstance(save_flag, bool), "The save_flag must be either True or False"
        assert (scale >= 0), "The scale is a positive number signifying the number of SDs in the inlier distribution"
        if strt_wvl is not None: assert isinstance(strt_wvl, (int, float)) and (strt_wvl < 2.602120), \
            "The starting wavelength must be a number less than 2.6 micrometer"
        if stop_wvl is not None: assert isinstance(stop_wvl, (int, float)) and (stop_wvl > strt_wvl), \
            "The stopping wavelength must be a number greater than the starting wavelength"
        assert isinstance(rem_noisy_pixel, bool), "The rem_noisy_pixel parameter must be a Boolean value"

        print('Processing the image {}\n'.format(os.path.basename(img_name)))

        'Read in the image'
        hsi_img, hsi_header = spectral_utilities().open_hsiImage(img_name, strt_wvl=strt_wvl, stop_wvl=stop_wvl)
        wvl = np.asarray(hsi_header["wavelength"], dtype=np.float32)

        'Read in the model image'
        mdl_name = img_name.replace('_nr_ds.img', '_mdl_ds.img')
        mdl_img, mdl_hdr = spectral_utilities().open_hsiImage(mdl_name, strt_wvl=strt_wvl, stop_wvl=stop_wvl)

        'Get the frame information'
        frm_info = spectral_utilities().extract_crismFrameInfo(hsi_img)

        'Estimate the continuum'
        hsi_cont = spectral_utilities().crism_contrem_mdl_nb(img_name, strt_wvl=strt_wvl, stop_wvl=stop_wvl)

        'Perform the continuum removal'
        hsi_crem = hsi_img / hsi_cont
        mdl_crem = mdl_img / hsi_cont

        'Apply spatial smoothing if needed'
        if filter_size is not None:
            if (filter_size >= hsi_crem.shape[0]) or (filter_size >= hsi_crem.shape[1]):
                print('Provided filter kernel is too big: SKIPPING SMOOTHING!!!!')
            else:
                'Apply spatial smoothing to each band'
                for ii in range(hsi_crem.shape[2]):
                    'Smooth the HSI image'
                    t1 = np.squeeze(hsi_crem[frm_info['strtRow']:frm_info['stopRow'],
                                    frm_info['strtCol']:frm_info['stopCol'], ii])
                    'Remove the Nan values and interpolate over them'
                    t1 = crism_processing().fill_nan(t1)

                    t1 = ndimage.uniform_filter(t1, size=filter_size)
                    hsi_crem[frm_info['strtRow']:frm_info['stopRow'],
                    frm_info['strtCol']:frm_info['stopCol'], ii] = t1

                    'Smooth the model image'
                    t2 = np.squeeze(mdl_crem[frm_info['strtRow']:frm_info['stopRow'],
                                    frm_info['strtCol']:frm_info['stopCol'], ii])
                    'Remove the Nan values and interpolate over them'
                    t2 = crism_processing().fill_nan(t2)

                    t2 = ndimage.uniform_filter(t2, size=filter_size)
                    mdl_crem[frm_info['strtRow']:frm_info['stopRow'],
                    frm_info['strtCol']:frm_info['stopCol'], ii] = t2

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

        temp_outname  = img_name.replace('.img', '_mask.hdr')
        envi.save_image(temp_outname, hsi_mask, force=True, interleave='bil')

        'Find the pixels with significant absorptions'
        idx_sigabs = np.where(hsi_mask == 1)

        if idx_sigabs[0].size != 0:
            'Get continuum removed spectra with significant absorptions'
            crspectra = hsi_crem[idx_sigabs[0], idx_sigabs[1], :]

            'Scale spectra if needed'
            #scl_lvl = 0.2
            if scl_lvl is not None:
                crspectra = spectral_utilities().scale_spectra_cr(crspectra, scale_lvl=scl_lvl)

            'Perform the open-set predictions for the significant spectra'
            spectra_pred = self.doc.predict_crism(np.expand_dims(crspectra, axis=2), scale=scale)

            'create a cube with the predicted data'
            doc_scores = np.zeros((hsi_crem.shape[0], hsi_crem.shape[1], spectra_pred.shape[1]))
            doc_scores[idx_sigabs[0], idx_sigabs[1], :] = spectra_pred

            'Save the image and associated quantities if needed'
            if save_flag:
                'Change the save folder if needed'
                if (self.base_folder is not None) and (self.target_folder is not None):
                    out_name = img_name.replace(self.base_folder, self.target_folder)
                out_name = img_name.replace('.img', '_doc_class_v2.hdr')
                out_name = out_name.replace('.IMG', '.hdr')
                'If the base folder to save the image does not exist create it as needed'
                if not os.path.isdir(os.path.dirname(out_name)):
                    os.makedirs(os.path.dirname(out_name))

                'Create a header for the doc classifier'
                header = {"Description": "Maps of the various outlier clusters identified automatically",
                          "key": "+ve - value identification, -1 - outlier"}
                header["lines"] = hsi_header["lines"]
                header["samples"] = hsi_header["samples"]
                header["bands"] = doc_scores.shape[2]
                if self.doc.class_names is not None:
                    header['band names'] = self.doc.class_names
                'Save the image'
                envi.save_image(out_name, doc_scores, dtype=np.float32, force=True, interleave='bil', metadata=header)

            'Get the ancilliary products'
            nvl_spct, nvl_spct_map = self.extract_significant_outliers(doc_scores, hsi_img, hsi_crem=hsi_crem,
                                                                       scl_lvl=scl_lvl)
            'If nothing is found- skip saving the ancilliary products'
            if nvl_spct is None or nvl_spct_map is None:
                save_anc_prdct = False

            if save_anc_prdct:
                'Header details'
                name_kmspectra = ['Est. Spectra- {:d}'.format(ii+1) for ii in range(nvl_spct.shape[0])]
                sli_hdr = {'wavelength': wvl,
                           'lines': nvl_spct.shape[0], 'samples': nvl_spct.shape[1], 'bands': 1,
                           'spectra names': name_kmspectra}
                sli_name = out_name.replace('.hdr', '_extracted_spectra')
                sli_ext_samp = sli(nvl_spct, sli_hdr, [])
                sli_ext_samp.save(sli_name)

                'Save the map of the extracted samples'
                header = {"Description": "The mineral identification maps for a known library",
                          "key": "+ve - value identification, -1 - outlier"}
                header["lines"] = hsi_header["lines"]
                header["samples"] = hsi_header["samples"]
                header["bands"] = nvl_spct_map.shape[2]
                header["band_names"] = name_kmspectra

                outmap_name = sli_name.replace('_extracted_spectra', '_extracted_spectra_map.hdr')
                envi.save_image(outmap_name, nvl_spct_map, dtype=np.float32, force=True,
                                interleave='bil', metadata=header)

            return doc_scores, nvl_spct, nvl_spct_map

        else:
            print('No significant spectra found!! Skipping!')
            return None, None, None

    def extract_significant_outliers(self, doc_map, hsi_cube, hsi_crem=None, scl_lvl=None):
        """
        This function is used identify/extract the significant outliers in a given image.

        :param doc_map: [ndarray: nRows X nCols X nMins]
        This cube contains the results of a Deep open classifier on the image spectra. Pixels with non-zero values in
        each band are identified as closest to that specific mineral. Futher, pixels with a value 2 in each band are
        flagged as outliers with respect to the mineral classes distribution.

        :param hsi_img: [ndarray: nRows X nCols X nBands]
        The 3D matrix which contains the original image.

        :param hsi_crem: [ndarray: nRows X nCols X nBands] (Default: None)
        The 3D matrix which contains the continuum removed image. If  no such image is given the continuum removed imge
        is calculated.

        :param scl_lvl: [0<= float <= 1](Default: None)
        The level to which the spectra are scaled before doing feature extraction
        :return:
        """
        assert hsi_cube.shape == hsi_crem.shape, "The I/F and continuum removed cubes must be of the same size"
        if scl_lvl is not None: assert (0 <= scl_lvl <= 1), "The scaling level must be between 0 & 1"

        if hsi_crem is None:
            hsi_cont = crism_processing().crism_contRem_nb(
                hsi_cube, frm_info= spectral_utilities().extract_crismFrameInfo(hsi_cube))
            hsi_crem = hsi_cube / hsi_cont

        'Initialize variable to hold the outliers'
        glbl_row_outliers = None
        glbl_col_outliers = None

        'Iterate over the minerals that have already been mapped'
        for min_num in range(doc_map.shape[2]):
            'Get the band corresponding to a specific mineral'
            min_band = np.squeeze(doc_map[:, :, min_num])

            'The outliers of this mineral class'
            [row_outliers, col_outliers] = np.where(min_band == 2)

            'Create the combined outliers'
            if glbl_row_outliers is None:
                glbl_row_outliers = row_outliers
                glbl_col_outliers = col_outliers
            else:
                glbl_row_outliers = np.hstack((glbl_row_outliers, row_outliers))
                glbl_col_outliers = np.hstack((glbl_col_outliers, col_outliers))

        'If there are not enough outliers no need to cluster'
        if (glbl_row_outliers.size == 0) or  (glbl_col_outliers.size == 0):
            return None, None

        'Get the outliers features'
        glbl_outliers_feat = self.doc.get_base_features(hsi_crem[glbl_row_outliers, glbl_col_outliers, :],
                                                        scl_lvl=scl_lvl)

        'Perform K-Means clustering over different values of k for the outliers'
        ssd = []
        for kk in tqdm(range(2, 18)):
            'Fit kmeans to the data'
            km = cluster.KMeans(n_clusters=kk)
            km = km.fit(glbl_outliers_feat)

            'Keep track of the sum of squared distances'
            ssd.append(km.inertia_)

        'Find the elbow point on the SSD to estimate the number of clusters'
        kn = KneeLocator(np.arange(2, 18), ssd, curve='convex', direction='decreasing',
                         interp_method='interp1d')
        num_clust = int(np.round(kn.knee * 1.5))

        print(('Based on the Knee there are {:d} clusters'.format(num_clust)))

        'Perform clustering with ideal number of clusters'
        km = cluster.KMeans(n_clusters=num_clust)
        km = km.fit(glbl_outliers_feat)
        km_labels = km.predict(glbl_outliers_feat)

        'Also estimate the class means from the image'
        class_mean_if = []
        class_mean_cr = []
        for ii in range(num_clust):
            idx = np.where(km_labels == ii)
            class_mean_if.append(np.nanmean(np.squeeze((hsi_cube[glbl_row_outliers,
                                                  glbl_col_outliers, :])[idx, :]), axis=0))
            class_mean_cr.append(np.nanmean(np.squeeze((hsi_crem[glbl_row_outliers,
                                                      glbl_col_outliers, :])[idx, :]), axis=0))

        'Find the distance between the class means in feature space'
        class_mean_if = np.asarray(class_mean_if)
        class_mean_cr = np.asarray(class_mean_cr)
        class_mean_feat = self.doc.get_base_features(class_mean_cr, scl_lvl=scl_lvl)

        'Prune the list of cluster means'
        final_class_means = []
        for ii in range(class_mean_feat.shape[0] - 1):
            dist = np.squeeze(1 - pairwise_distances(class_mean_feat[ii, :].reshape(1, -1),
                                                     class_mean_feat, metric='cosine'))
            dist[:(ii+1)] = 0
            'If there are no similar exemplars add this exemplar to the list'
            if np.max(dist )< 0.95:
                final_class_means.append(class_mean_if[ii, :])
            else:
                'If there is a similar class merge this class into the closest class'
                mg_class = np.argmax(dist)
                'Update the new class mean'
                oc_n = len(np.where(km_labels == ii)[0])
                mc_n = len(np.where(km_labels == mg_class)[0])
                class_mean_if[mg_class, :] = ((oc_n * class_mean_if[ii, :]) + (mc_n *
                                                                          class_mean_if[mg_class, :])) / (oc_n + mc_n)
                'Update class labels'
                km_labels[np.where(km_labels == ii)[0]] = mg_class

        final_class_means.append(class_mean_if[-1, :])
        final_class_means = np.asarray(final_class_means)

        'Convert labels into label map- showing outlier position in the image'
        km_label_map = np.zeros((hsi_cube.shape[0], hsi_cube.shape[1], len(np.unique(km_labels))))
        ctr = 0
        for ii in np.unique(km_labels):
            idx = np.where(km_labels == ii)[0]
            km_label_map[glbl_row_outliers[idx], glbl_col_outliers[idx], ctr] = 1
            ctr += 1


        return final_class_means, km_label_map

    def map_folder(self, folder_loc, img_string=".img", scl_lvl=None, filter_size=None, scale=1., strt_wvl=None,
                          stop_wvl=None, save_anc_prdct=False, save_flag=True):
        """
        This function can be used to iterate over all the files inside the folder location provided and maps images
        containing specific substrings. The images are mapped using a DOC mapping object.

        :param folder_loc: (string)
        The string which is the address of a physical location which contains a bunch of images of the type we want to
        process.

        :param img_string (string) (Default:'.img')
        This argument provides an additional mode of filtering the CRISM images in the specific folder. In this case it
        only considers image which consists of the sting provided in this argument.

        :param scl_lvl (float) (Default: None)
        This level to which a spectrum is scaled to after continuum removal.

        :param filter_size (int) (Default: None)
        The size of spatial filter kernel applied to the data.

        :param strt_wvl [float] (Default: None)
        The wavelength from which the image is being read. If None is given starts from the first band.

        :param stop_wvl [float] (Default: None)
        The wavelength upto which the image is being read. If None is given goes till band.

        :return: None
        """

        assert os.path.isdir(folder_loc), "The specified folder does not exist"
        assert isinstance(img_string, str), "The sli_string must be a string variable"

        'Iterate over the files in the specified folder'
        for r, d, f in os.walk(folder_loc):
            for file in f:
                if file.find(img_string) != -1:
                    'The image address is'
                    img_address = os.path.join(r, file)

                    if "FRT" in img_address:
                        filter_size = 5
                    else:
                        filter_size = 3

                    'Map the image by using the DOC object'
                    _, _, _ = self.crism_doc_mapping(img_name=img_address, scl_lvl=scl_lvl, scale=scale,
                                                  filter_size=filter_size, save_flag=save_flag, strt_wvl=strt_wvl,
                                                  stop_wvl=stop_wvl, save_anc_prdct=save_anc_prdct)




    """'Provide an image name'
    img_name = os.path.join('/Volume1/data/CRISM/atmCorr_stable/oxia_planum',
                            'FRT00004686', 'FRT00004686_07_IF166L_TRRB_sabcondpub_v1_nr_ds.img')
    'Create a DOC mapping object'
    obj1 = crism_mapping()
    _, _, _ = obj1.crism_doc_mapping(img_name=img_name, scl_lvl=0.2, filter_size=5, save_flag=True, strt_wvl=1.0275,
                                     save_anc_prdct=True)"""

    'Set the folder to be processed'
    folder_loc = '/Volume1/data/CRISM/atmCorr_stable/mawrth_vallis/FRT0000A425/'

    'Create a DOC mapping object'
    obj1 = crism_mapping()
    obj1.map_folder(folder_loc=folder_loc, img_string='_nr_ds.img', scl_lvl=0.2, strt_wvl=1.0275, stop_wvl=2.6,
                    save_anc_prdct=True, save_flag=True)

