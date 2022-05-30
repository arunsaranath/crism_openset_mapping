# -*- coding: utf-8 -*-

"""
FileName:               crism_processing
Author Name:            Arun M Saranathan
Description:            This code file is used to extract samples the CRISM Images where the MICA samples were
                        originally found. A simple heurestic is used to eliminate spectra which appear generally flat so
                        that they do not affect the model training.
Dependancies:           numba, spectral

Date Created:           19th June 2021
Last Modified:          19th June 2021
"""

'Import python core packages'
import spectral.io.envi as envi
import numpy as np
from tqdm import tqdm
from scipy import interpolate
from scipy.ndimage.filters import uniform_filter1d as filter1d
import numba as nb
import os


@nb.jit("f4[:, :](f4[:], f4[:])")
def convex_hull_jit(wvl, spectrum):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements the algorithm CONVEXHULL(P) described by  Mark de Berg, Otfried
    Cheong, Marc van Kreveld, and Mark Overmars, in Computational Geometry:
    Algorithm and Applications, pp. 6-7 in Chapter 1

    :param points: A N X 2 matrix with the wavelengths as the first column
    :return: The convex hull vector
    """
    'The starting points be the first two points'
    xcnt, y = wvl[:2], spectrum[:2]
    'Now iterate over the other points'
    for ii in range(2, spectrum.shape[0], 1):
        'check next prospective convex hull members'
        xcnt = np.append(xcnt, wvl[ii])
        y = np.append(y, spectrum[ii])
        flag = True

        while (flag == True):
            'Check if a left turn occurs at the central member'
            a1 = (y[-2] - y[-3]) / (xcnt[-2] - xcnt[-3])
            a2 = (y[-1] - y[-2]) / (xcnt[-1] - xcnt[-2])
            if (a2 > a1):
                xcnt[-2] = xcnt[-1]
                xcnt = xcnt[:-1]
                y[-2] = y[-1]
                y = y[:-1]
                flag = (xcnt.shape[0] > 2);
            else:
                flag = False

    return np.vstack((xcnt, y))


class crism_processing(object):
    def crism_contRem_nb(self, img_cube, wvl=None, frm_info=None, filter_size=None):
        """
        This function is used perform continuum removal on a specific image.

        :param img_cube: [ndarray: nRows X nCols X nBands]
        The image cube on which we want to perform continuum removal

        :param wvl: [narray: nBands] (Default: None)
        The wavelength vector associated with the image cube. If none is provided then an integer will be assigned to
        each band

        :param frm_info: [dict] (Default: None)
        A dictionary with the start and stop rows and columns.

        :param filter_size: [int] (Default: None)
        If an integer values is given a filter of that size is applied to the spectra before continuum removal

        :return: [ndarray: nRows X nCols X nBands]
        A cube which contains the continuum spectrum associated with each pixel
        """
        'Check the input arguments to the function'
        assert (len(img_cube.shape) == 3), "The image cube must be 3D"
        if wvl is None:
            wvl = np.arange(img_cube.shape[2])
        assert (len(wvl) == img_cube.shape[2]), "The size of the wavelength vector must be equal to the number of bands"
        if frm_info is None:
            frm_info = {"strtRow": 0, "stopRow": img_cube.shape[0], "strtCol": 0, "stopCol": img_cube.shape[1]}
        assert isinstance(frm_info, dict)
        if filter_size is not None:
            assert isinstance(filter_size, int) and (filter_size >= 0), "The filter size must be a positive integer"
            assert (filter_size <= img_cube.shape[2]), "The size of the filter must be less than the number of bands " \
                                                       "in the image"

        'Create a matrix to hold the background'
        cube_bg = np.empty(img_cube.shape, dtype=np.float32)
        cube_bg[:] = np.nan

        'For each pixel find the continuum'
        for ii in tqdm(range(frm_info["strtRow"], frm_info["stopRow"]+1)):
            for jj in range(frm_info["strtCol"], frm_info["stopCol"]+1):
                'The spectrum is'
                spectrum = np.squeeze(img_cube[ii, jj, :])
                if filter_size is not None:
                    spectrum = filter1d(spectrum, filter_size)

                'Check if it has nans'
                flag = np.isnan(spectrum).any()

                'if not nan find the continuum'
                if not flag:
                    'Calculate the convex hull'
                    cHull = convex_hull_jit(wvl, spectrum)

                    f = interpolate.interp1d(np.squeeze(cHull[0, :]), np.squeeze(cHull[1, :]))
                    ycnt = f(wvl)

                    'Place this continnum in the folder'
                    cube_bg[ii, jj, :] = ycnt

        return cube_bg

    def matrix_contrem_nb(self, matrix, wvl=None, filter_size=None):
        """
        This function can be used perform continuum removal on each row of a matrix.

        [N.B.: The function  will skip processing a row if it contains any NaNs]

        :param matrix: [ndarray: nRows X nBands]
        The matrix on which we want to perform continuum removal

        param wvl: [narray: nBands](Default: None)
        The wavelength vector associated with the image cube.

        :param filter_size: [int] (Default: None)
        If an integer values is given a filter of that size is applied to the spectra before continuum removal

        :return: [ndarray: nRows X nBands]
        A matrix which contains the continuum spectrum associated with each pixel
        """

        assert isinstance(matrix, np.ndarray), "The <matrix> varible must be a numpy matrix"
        assert len(matrix) == 2, "The <matrix> vavrible must be a 2d array"
        if wvl is None:
            wvl = np.arange(0, matrix.shape[1])
        assert wvl.shape[0] == matrix.shape[1], "The <wvl> variable must have the same dimensions as each " \
                                                "rows of <matrix>"


        'Create a matrix to hold the background'
        matrix_bg = np.empty(matrix.shape, dtype=np.float32)
        matrix_bg[:] = np.nan

        'For each pixel find the continuum'
        for ii in tqdm(range(matrix.shape[0])):
            'The spectrum is'
            spectrum = np.squeeze(matrix[ii, :])
            if filter_size is not None:
                spectrum = filter1d(spectrum, filter_size)

            'Check if it has nans'
            flag = np.isnan(spectrum).any()

            'if not nan find the continuum'
            if not flag:
                'Calculate the convex hull'
                cHull = convex_hull_jit(wvl, spectrum)

                f = interpolate.interp1d(np.squeeze(cHull[0, :]), np.squeeze(cHull[1, :]))
                ycnt = f(wvl)

                'Place this continnum in the folder'
                matrix_bg[ii, :] = ycnt

        return matrix_bg

    def crism_contrem_mask(self, cr_img_cube, flat_thresh=0.02):
        """
        A spectrum is considered significant if it contains one absorption which is greater than a specific percentage
        of the continuum, i.e. absorptions which are smaller than a specific threshold are considered insignificant.
        This function basically only finds a mask which identifies spectra in a image such that its minimum value is
        below a specific threshold.

        :param cr_img_cube: [ndarray: nRows X nCols X nBands]
        The image cube after continuum removal

        :param flat_thresh: [float] (Default: 0.98)
        This parameter is 1-threshold which indicated significance.

        :return: mask [ndarray: nRows X nCols]
        A pixel value is 1 if it has a significant absorption else 0
        """

        assert (0 <= flat_thresh <= 1), "the significant threshold must be a percentage"
        #assert (np.any(np.abs(cr_img_cube[:,:, 1] - 1) > 0.1)), "The image should be continuum removed"

        'Find the samllest value in each continuum removed spectrum'
        cr_img_mask = np.nan_to_num(np.min(cr_img_cube, axis=2))

        'Apply the threshold for significant absorptions'
        cr_img_mask[cr_img_mask >= (1 - flat_thresh)] = 0
        cr_img_mask[cr_img_mask != 0] = 1

        return cr_img_mask

    def crism_fillnan(self, img_cube):
        """
        interpolate to fill nan values based on other entries in the neighborhood of the pixels

        :param cr_img_cube: [ndarray: nRows X nCols X nBands]
        The image cube after continuum removal

        :return: the matrix with the nan interpolated by the nan values
        """

        assert len(img_cube.shape) == 3, "The image cube must have 3 dimensions"

        'Get the image shape and convert it into an array'
        [rows, cols, bands] = img_cube.shape
        arr_img = img_cube.reshape((rows * cols, bands))

        'Replace the NaNs with interpolated values'
        arr_img = self.fill_nan(arr_img)

        'Reshape to image size'
        img_cube = arr_img.reshape((rows, cols, bands))

        return img_cube

    def fill_nan(self, data):

        """
        interpolate to fill nan values based on other entries in the row

        :param data: [nRows X nCols]
        a numpy matrix with nan values

        :return: the matrix with the nan interpolated by the nan values
        """
        assert len(data.shape) ==2, "The data variable must be a 2D matrix"

        ok = ~np.isnan(data)
        xp = ok.ravel().nonzero()[0]
        fp = data[~np.isnan(data)]
        x = np.isnan(data).ravel().nonzero()[0]
        data[np.isnan(data)] = np.interp(x, xp, fp)
        return data




