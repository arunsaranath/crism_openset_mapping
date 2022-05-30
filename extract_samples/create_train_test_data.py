# -*- coding: utf-8 -*-

"""
FileName:               generate_basemodel_labels
Author Name:            Arun M Saranathan
Description:            This code file is used to add the labels associated with the clearest identifications of each
                        mineral sample corresponding to known classes. This specific code file generates the labels
                        based base GAN model

Date Created:           21st June 2021
Last Modified:          23rd June 2021
"""

'Import core libraries'
import numpy as np
import pandas as pd
import os
import re
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from scipy.stats import dirichlet

from spectral_utilities.spectral_utilities import spectral_utilities
from crism_processing.crism_processing import crism_processing
from hsiUtilities.crism_ganProc_utils import crism_ganProc_utils

class create_train_test_data(object):
    def __init__(self, store_loc, spectra_table="spectral_data", cospectra_table="cospectral_data"):
        """
        This class is used to generate the labels for samples in a specific pandas datastore. This class is built for
        the scenario the store has tables for both the spectral data and another table for its corresponding spectra.

        :param store_loc: (string)
        This parameter is a string variable which is the physical address of the pandas HDFStore

        :param spectra_table: (string) (Default: "spectral_data")
        The name of the table which contains the spectral data.

        :param cospectra_table: (string) (Default: "spectral_data")
        The name of the table which contains the continuum corresponding to the spectral data.
        """

        'Basic checks on the provided parameters'
        assert os.path.isfile(store_loc), "This pandas HDFStore does not exist"
        assert isinstance(spectra_table, str), "The table name must be a string"
        assert isinstance(cospectra_table, str), "The table name must be a string"

        'Initialize a variable to hold the labels'
        self.spectra_labels = pd.DataFrame([])
        self.store_loc = store_loc

        'Get the spectral and continuum data from the pandas HDFStore'
        store = pd.HDFStore(self.store_loc, "r")
        self.spectra_df = store.get(spectra_table)
        self.cospectra_df = store.get(cospectra_table)
        'Close the store'
        store.close()

        'Now also get the wavelength from the pandas dataframe'
        wvl = [re.findall("[+-]?\d+\.\d+", val) for val in self.spectra_df.columns]
        self.wvl = np.squeeze(np.asarray(wvl, dtype=np.float32))

    def generate_labels_basemodel(self, model, exemplar_sli, scl_lvl=None, batch_size=1000, label_set_name=None
                                  , endmem_set_name=None, thresh=0.955):
        """
        This function is used generate the labels for the spectra in a specific store.

        :param model: (keras model)
        This is a model that can be used to compute the features of the spectra in the pandas HDFStore

        :param exemplar_sli: (string)
        The spectral library which contains the exemplars for which we are using as part of the labels

        :param scl_lvl: (0 < float 1) (Default: None)
        A float variable that indicates the size of the largest absorption band in the spectrum. If no value is provided
        (default) no scaling is done. If it is provided the size of the largest absorption is equal to sclLvl. This
        parameter is in (0, 1]

        :param batch_size (int) (Default: 1000)
        This function performs the labeling incrementally. This parameter is the batch-size used for the incremental
        labelling. Default is 1000

        :param label_set_name (string) (Default: None)
        If this string parameter is provided the labels are added to the store in a table of this name.

        :param endmem_set_name (string) (Default: None)
        If this string parameter is provided the endmember names are added to the store in a table of this name.

        :param thresh (0 <= thresh <=1) (Default: 0.975)
        The cosine value above which a samples is considered as the same as the exemplar.

        :return: None
        This function updates the store to contain the label information and the name of the endmembers etc..
        """

        assert isinstance(model, tf.keras.Model), "The representation model must be a keras model"
        assert os.path.isfile(exemplar_sli), "The spectral library with the exemplars does not exist"
        assert (scl_lvl is None) or (0 <= scl_lvl <= 1), "Must be a number between 0 & 1"
        assert isinstance(batch_size, int) and (batch_size < self.spectra_df.shape[0]), \
            "Must be a integer smaller than number of data samples"
        if label_set_name is None:
            label_set_name = 'labels'
        assert isinstance(label_set_name, str), "The table name for the labels must be a string"

        if endmem_set_name is None:
            endmem_set_name = "endmembers"
        assert isinstance(endmem_set_name, str), "The table name for the tables with the endmembers must be a string"

        'If the a table with expected label name already exists first remove it and redo the labeling'
        store = pd.HDFStore(self.store_loc, "a")
        if ('/' + label_set_name) in store.keys():
            store.remove(label_set_name)

        'Similarly check if the table with endmember names exist'
        if ('/' + endmem_set_name) in store.keys():
            store.remove(endmem_set_name)
        store.close()

        'Step 1: Preprocess the exemplar spectra'
        if scl_lvl is not None:
            exemplar_crspectra, hdr = spectral_utilities().prepare_slifiles(exemplar_sli, strt_wvl=1.0275,
                                                                            cont_rem=True, scl_lvl=scl_lvl)
        else:
            exemplar_crspectra, hdr = spectral_utilities().prepare_slifiles(exemplar_sli, strt_wvl=1.0275,
                                                                            cont_rem=True)

        'Get the end-member names'
        endmem_names = hdr['spectra names']

        'Reshape the data to be suitable to model'
        exemplar_crspectra = np.expand_dims(exemplar_crspectra, axis=2)

        'Perform feature extraction on the exemplars'
        exemplar_feat = model.predict(exemplar_crspectra)

        'Iterate over samples in the spectral dataframe'
        for ii in tqdm(range((self.spectra_df.shape[0] // batch_size) + 1)):
            strt_indx, stop_indx = (ii*batch_size), ((ii+1) * batch_size)
            if stop_indx >= self.spectra_df.shape[0]:
                stop_indx = self.spectra_df.shape[0]
            'Get a batch of spectral data'
            batch_spectra = np.asarray(self.spectra_df.iloc[strt_indx:stop_indx, :])
            batch_cospectra = np.asarray(self.cospectra_df.iloc[strt_indx:stop_indx, :])

            'Get the continuum removed spectra for the batch'
            batch_crspectra = batch_spectra / batch_cospectra

            'Normalize the spectra in the batch'
            if scl_lvl is not None:
                batch_crspectra = spectral_utilities().scale_spectra_cr(batch_crspectra, scale_lvl=scl_lvl)

            'Expand dimension to make suitable to apply to the model'
            batch_crspectra = np.expand_dims(batch_crspectra, axis=2)

            'Perform feature extraction on the exemplars'
            batch_feat = model.predict(batch_crspectra)

            'Find the pairwise cosine similarity between the exemplars and the batch spectra'
            batch_sim = cosine_similarity(exemplar_feat, batch_feat)
            'Threshold the similarity matrix to only preserve the largest value'
            batch_sim[batch_sim <= thresh] = 0

            'Get the labels for this batch'
            labels = -1 * np.ones((batch_spectra.shape[0],), dtype=np.int)
            'Update the labels for a sample if its similarity to the closest exemplar is greater than 0.95'
            labels[np.max(batch_sim, axis=0) >= thresh] = np.argmax(batch_sim[:, np.max(batch_sim, axis=0) >= thresh],
                                                                  axis=0)

            'Update the data-frame holding the labels'
            self.spectra_labels = self.spectra_labels.append(pd.DataFrame(labels))

        'If needed add the labels to the store'
        if label_set_name is not None:
            'Set the column name for spectra_labels'
            self.spectra_labels.columns = ['Labels']
            'Save the labels into the pandas HDFStore'
            store = pd.HDFStore(self.store_loc, "a")
            'Now save the data we have extracted'
            store.append(label_set_name, self.spectra_labels, format='table', data_columns=True)
            store.append(endmem_set_name, pd.DataFrame(endmem_names, columns=['end_members']), format='table',
                         data_columns=True)
            'Close the store'
            store.close()

        return self.spectra_labels

    def get_classnames(self, table_name):
        """
        This function will check the store to see if there is a table which contain the class names. If it exists it
        will return the names of these classes

        :param table_name: [string]
        The name of the table is the store

        :return:
        A list with the names of the classes
        """

        with pd.HDFStore(self.store_loc) as hdf:
            'Check that the table of your choice exists'
            assert ('/' + table_name) in hdf.keys()

            return np.asarray(hdf.select(table_name))

    def get_train_test(self, label_set_name, test_frac=0.15, n_samp=None, scl_lvl=None, aug_data=True):
        """
        This function will look at the data in the store and based on the labels in the store partitions the data into
        training and testing samples. This function only partitions the positively labeled samples and does not consider
        unlabeled samples.

        :param label_set_name (string)
        This is the name of the table which contains the labels to be used to partition the data into test and training
        samples

        :param test_frac: (0 <= float <= 1) (Default: 0.15)
        Splits samples from each class into a training and testing set. The fraction of test samples is decided by this
        parameter. Default is 0.15 or 15%

        :param n_samp (int) (Default: None)
        If provided this number of samples from each class are split into the training and test set. I use this to
        sub-sample and rebalance the dataset

        :param scl_lvl: (0 < float 1) (Default: None)
        A float variable that indicates the size of the largest absorption band in the spectrum. If no value is provided
        (default) no scaling is done. If it is provided the size of the largest absorption is equal to sclLvl. This
        parameter is in (0, 1]

        :param aug_data: [Boolean] (Default: True)
        If used the classes with lesser samples than the number of samples are augmented to make the class appear
        bigger.

        :return: (tuple)
        A tuple which in order contains (x_train, x_test, y_train, y_test) where each is a numpy array
        """

        assert (0 <= test_frac <= 1), "This test_frac must be a value between 0 & 1"
        assert (n_samp is None) or (isinstance(n_samp, int) and (n_samp > 0)), "This value must be a positive integer"
        assert (scl_lvl is None) or (0 <= scl_lvl <= 1), "Must be a number between 0 & 1"

        'Initialize variable to hold the training and testing values'
        x_train, x_test, y_train, y_test = None, None, None, None

        'Get the labels from the store'
        store = pd.HDFStore(self.store_loc, "r")
        assert label_set_name in store, "The requested table does not exist"
        labels = store.get(label_set_name)
        labels = np.asarray(labels)
        store.close()

        'Iterate over the labels'
        for ii in tqdm(range(int(labels.max() + 1))):
            'Get the indicies of the class and randomly shuffle them'
            class_idx = np.where(labels == ii)[0]
            random.shuffle(class_idx)
            'Reduce the selection if needed'
            if (n_samp is not None) and (len(class_idx) > n_samp):
                class_idx = class_idx[:n_samp]

            'Get the samples corresponding to training and test samples'
            n_train = int((1. - test_frac) * len(class_idx))
            class_samp = np.asarray(self.spectra_df.iloc[class_idx, :]) / \
                               np.asarray(self.cospectra_df.iloc[class_idx, :])

            'Augment the data if necessary'
            if (n_samp is not None) and (len(class_idx) < n_samp) and (aug_data):
                'Create a convex combinations of the rows to augment the data'
                alpha = np.abs(np.random.normal(1, 0.001, 3))
                alpha[0] = 20
                'Sample weights from this dirichlet'
                final_weights = np.zeros((n_samp - class_samp.shape[0], class_samp.shape[0]))
                final_weights[:, :3] = dirichlet.rvs(alpha, size=n_samp - class_samp.shape[0])

                'Now shuffle the weights'
                def crazy_shuffle(arr):
                    """
                    This function is used to randomly shuffle the columns of a matrix

                    :param arr: [ndarray:]
                    This is a 2D numpy array.
                    :return:
                    """
                    x, y = arr.shape

                    rows = np.indices((x, y))[0]
                    cols = [np.random.permutation(y) for _ in range(x)]
                    return arr[rows, cols]

                final_weights = crazy_shuffle(final_weights)
                'Get the augmented class matrix'
                aug_spectra = np.matmul(final_weights, class_samp)
                class_samp = np.vstack((class_samp, aug_spectra))
                'Update the number of training samples based on the augmented matrix'
                n_train = int((1. - test_frac) * class_samp.shape[0])
                #class_idx = np.arange(n_train)

            'Create class labels'
            class_labels = np.asarray([ii] * class_samp.shape[0])
            'Normalize the spectra in this class'
            if scl_lvl is not None:
                class_samp = spectral_utilities().scale_spectra_cr(class_samp, scale_lvl=scl_lvl)

            "Split into training and testing data"
            class_train_samp = class_samp[:n_train, :]
            class_train_labels = class_labels[:n_train]
            class_test_samp = class_samp[n_train:, :]
            class_test_labels = class_labels[n_train:]

            """'Class Training Labels'
            class_train_labels = class_labels[:n_train]
            class_test_samp  = np.asarray(self.spectra_df.iloc[class_idx[n_train:], :]) / np.asarray(
                self.cospectra_df.iloc[class_idx[n_train:], :])
            'Normalize the spectra in this class'
            if scl_lvl is not None:
                class_test_samp = spectral_utilities().scale_spectra_cr(class_test_samp, scale_lvl=scl_lvl)

            'Class Test Labels'
            class_test_labels = class_labels[n_train:]"""

            'Add this to our data variables'
            if x_train is None:
                x_train, x_test, y_train, y_test = class_train_samp, class_test_samp, class_train_labels, \
                                                   class_test_labels
            else:
                x_train, x_test, y_train, y_test = np.vstack((x_train, class_train_samp)), \
                                                   np.vstack((x_test, class_test_samp)), \
                                                   np.hstack((y_train, class_train_labels)), \
                                                   np.hstack((y_test, class_test_labels))

        return (x_train, x_test, y_train, y_test)

    def get_openset(self, label_set_name, n_samp= 10000, scl_lvl=None):
        """
        This function is used to get the samples in the store that are adjudged as being in the open store.

        :param label_set_name (string)
        This is the name of the table which contains the labels to be used to partition the data into test and training
        samples

        :param n_samp: (int) (Default: 10000)
        The number of open set samples that are needed in this case.

        :return:
        """

        assert (n_samp is None) or (isinstance(n_samp, int) and (n_samp > 0)), "This value must be a positive integer"
        assert (scl_lvl is None) or (0 <= scl_lvl <= 1), "Must be a number between 0 & 1"

        'Get the labels from the store'
        store = pd.HDFStore(self.store_loc, "r")
        assert label_set_name in store, "The requested table does not exist"
        labels = store.get(label_set_name)
        store.close()
        labels = np.asarray(labels)


        'Get all the open-set points'
        class_idx = np.where(labels == -1)[0]
        random.shuffle(class_idx)

        'Get the open set samples'
        x_openset = np.asarray(self.spectra_df.iloc[class_idx[:n_samp], :]) / np.asarray(
                self.cospectra_df.iloc[class_idx[:n_samp], :])
        'Normalize the spectra if needed'
        if scl_lvl is not None:
            x_openset = spectral_utilities().scale_spectra_cr(x_openset, scale_lvl=scl_lvl)

        y_openset = np.asarray(n_samp * [-1])

        return x_openset, y_openset


if __name__ == "__main__":
    'Intialize the HDFStore location'
    store_loc = '/Volume2/arunFiles/CRISM_minMapping_Ident/python_codeFiles/extract_samples/data_stores/mica_images.h5'

    '------------------------------------------------------------------------------------------------------------------'
    'LABEL CREATION'
    'Open and get the exemplar spectra'
    sli_name = os.path.join('/Volume2/arunFiles/CRISM_minMapping_Ident/python_codeFiles',
                            'mica_processing/data_products/exemplars_r1.sli')

    'Create the feature extractor of interest'
    disRep = crism_ganProc_utils().create_rep_model()

    'Get the location of significant spectra from the mica images'
    #_ = create_train_test_data(store_loc).generate_labels_basemodel(disRep, sli_name, scl_lvl=0.2,
    #                                                                label_set_name='labels')
    '------------------------------------------------------------------------------------------------------------------'
    'EXTRACT TRAINING AND TEST DATA'
    (xtr, xte, ytr, yte) = create_train_test_data(store_loc).get_train_test('labels', n_samp= 2500, scl_lvl=0.2)
    #xo, yo = create_train_test_data(store_loc).get_openset('labels', n_samp=5000, scl_lvl=0.2)

    print('Number of classes: {:d}'.format(max(ytr) + 1))

    '------------------------------------------------------------------------------------------------------------------'
    'GET THE CLASS NAMES IF ANY'
    #class_names = create_train_test_data(store_loc).get_classnames("endmembers")
