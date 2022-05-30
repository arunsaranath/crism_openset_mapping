# -*- coding: utf-8 -*-

"""
FileName:               extract_crism_outliers
Author Name:            Arun M Saranathan
Description:            This code file is used to extract spectra from the CRISM images which are considered as outliers
                        with respect known endmembers. Interesting/Relevant spectra which are not close to any of the
                        known endmembers are placed in an unknown class.

Dependancies:           numpy, os, spectral_utilities

Date Created:           12th July 2021
Last Modified:          12th July 2021
"""

'import core libraries'
import numpy as np
import pandas as pd
import os

'User defined libraries'
from spectral_utilities.spectral_utilities import spectral_utilities

class extract_crism_outliers(object):
    def __init__(self, store_name, n_cols, col_names=None, append_mode=True):
        """
        The constructor can be used to create a pandas HDFStore with a specific architecture.

        :param store_name: (string)
        This variable encodes the name of pandas HDFStore which contains the data which we are dealing with.

        :param n_cols: (int)
        The number of columns in the tables present in the store.

        :param col_names: (list: string) [Default: None]
        The names of the columns in the pandas dataframes. If  no names are provided the columns are named 'Col-1',
        'Col-2' and so on.

        :param append_mode: (Boolean) [Default: True]
        The samples are appended to existing tables if any. If the table of the specified name does not exist a new
        table of that name is created. If the table exists the data is appended to table of the specific name.
        """

        assert os.path.exists(store_name), "The provided string must be a valid address"
        assert isinstance(n_cols, int) and (n_cols > 0), "The number of columns must be a positive integer"
        if col_names is not None:
            assert [isinstance(item, str) for item in col_names], "The columns names must be a list of strings"
            assert len(col_name) == n_cols, "There must be a column name for each column"

        'Keep track of the variables needed'
        self.store_name = store_name
        self.n_cols = n_cols
        'Get the names of the tables in the store'
        store = pd.HDFStore(self.store_name, 'r')
        self.table_names = store.keys()
        if col_names is None:
            self.col_names = ['Col-{:d}'.format(ii) for ii in range(n_cols)]
        else:
            self.col_names = col_names

    def insert_samples2Datastores(self, table_name, sample_data):
        """
        This function can be used add samples to specific tables in a training store. If a table does not exist in the
        store a new table is created in the store. If the 'append_mode' is set to False, the function replaces the table
        in the store with new table with the samples provided in the function.

        :param table_name: (string)
        The name of the table in which the data provided to the functions is to be added. The exact nature of how the
        data interacts with the table depends on the 'append_mode' of the object.

        :param sample_data: (ndarray: n_samples X n_cols)
        A 2D matrix with the number of columns equal to the 'n_cols', and the rows correspond to the samples.

        :return:None
        """

        assert isinstance(table_name, str), "The table name must be a string"
        assert (len(sample_data.shape)==2) and (sample_data.shape[1] == self.n_cols), "The variable must be a 2D " \
                                                                                      "matrix- with the number of " \
                                                                                      "columns equal to the value" \
                                                                                      "fixed in the object"

        'Check if the table exists in a store'
        if ("/" + table_name) not in self.table_names:
            'If table does not exist add the samples as new table'
            store = pd.HDFStore(store_name, "w")
            'Now save the data we have extracted'
            store.put(table_name, pd.DataFrame(sample_data, columns=self.col_names), format='table', append= True,
                         data_columns=True)
            store.close()

            'Add this new table to the list of existing tables'
            self.table_names.append("/" + table_name)
        else:
            'If table exists we need to append the samples to an existing table'
            store = pd.HDFStore(store_name, "a")
            'Now save the data we have extracted'
            store.append(table_name, pd.DataFrame(sample_data, columns=self.col_names), format='table',
                         data_columns=True)
            store.close()



