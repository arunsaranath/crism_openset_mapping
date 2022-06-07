# -*- coding: utf-8 -*-

"""
FileName:               distance_based_clustering
Author Name:            Arun M Saranathan
Description:            This code file is used to produce a clustering in the feature space generated by the discrimi-
                        nator. It used a procedure similar to the Felzenszwalb segmentation, wherein it first creates
                        a 8-connected graph where each sample is connected to its 8-nearest neighbors and merges points
                        if it's addition will not change the MST of the combined segment over a specified threshold

                        [1] Saranathan, Arun M., and Mario Parente. "Uniformity-based superpixel segmentation of hyper-
                        spectral images." IEEE Transactions on Geoscience and Remote Sensing 54, no. 3 (2015):
                        1419-1430.

Dependancies:           numpy, spectral, spectral_utilities, hsiUtilities

Date Created:           18th May 2022
Last Modified:          18th May 2022
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from spectral_utilities.spectral_utilities import spectral_utilities
from hsiUtilities.crism_ganProc_utils import crism_ganProc_utils
from fastdist import fastdist

SUPPORTED_DISTANCES = ['cosine']
SUPPORTED_SMALLSEG_MODES = ['merge', 'outlier']

def top_k(arr, k, axis = 0):
    """
    This function be used to zero out all but the top-num_nn values in a numoy array

    :param arr: (np.ndarray)
    A numpy array which is to be thresholded to preserve the top-num_nn values along an axis

    :param k: (int)
    The number of values to be preserved along the specific dimension

    :param axis: (int) [Default: 0]
    The dimension along which the thresholding is to be done


    :return: out (np.ndarray)
    The array thresholded along specific dimension
    """
    assert isinstance(arr, np.ndarray), "The <arr> variable must be a numpy array"
    assert 0<= axis< len(arr.shape), f"The varible arr is {len(arr.shape)}-D, choose axis between (0-{len(arr.shape)})"
    assert isinstance(k, int), "The <num_nn> variable must be an integer"
    assert 0 < k < arr.shape[axis], f"The array has {arr.shape[axis]} elements in chosen axis, choose num_nn between " \
                                    f"(0-{arr.shape[axis]})"

    'indices of top num_nn values in axis'
    top_k_idx= np.take_along_axis(np.argpartition(arr, -k, axis= axis), np.arange(-k, -1), axis= axis)
    'create zero array'
    out= np.zeros.like(arr)
    'put idx values of arr in out'
    np.put_along_axis(out, top_k_idx, np.take_along_axis(arr, top_k_idx, axis= axis), axis= axis)
    return out

#def cosine_similarity_old(A):
    """
    This function can be used to find the cosine similarity between the different rows

    :param A:
    :return:
    """

    """'replace this with A.dot(A.T).toarray() for sparse representation'
    similarity = np.dot(A, A.T)

    'squared magnitude of preference vectors (number of occurrences)'
    square_mag = np.diag(similarity)

    'inverse squared magnitude'
    inv_square_mag = 1 / square_mag

    'if it doesnt occur, set its inverse magnitude to zero (instead of inf)'
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    'inverse of the magnitude'
    inv_mag = np.sqrt(inv_square_mag)

    'cosine similarity (elementwise multiply by inverse magnitudes)'
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag

    return cosine"""

def cosine_similarity_chunk(full_mat, batch_size=1000):
    """
    This function can be used to calculate the pairwise distances between the rows of a matrix. It will
    calculate these distances chunk by chunk

    :param full_mat:[np.ndarray: nSamples X nFeatures]


    :return:
    """
    assert isinstance(full_mat, np.ndarray), "The <full_mat> variable must be a numpy array"
    assert len(full_mat.shape)==2, "The <full_mat> variable must be a 2D array"
    assert isinstance(batch_size, int), "The <batch_size> variable must be an integer"
    assert 0 < batch_size <= full_mat.shape[0], "The <batch_size> variable must be a positice integer" \
                                                 "smaller than the number of samples"

    'Estimate the number of batches for the matrix'
    n_batches = (full_mat.shape[0] // batch_size) + 1

    'Initialize a distance matrix to hold the pairwise distances'
    distances = np.zeros((full_mat.shape[0], full_mat.shape[0]))

    for ii in tqdm(range(n_batches)):
        'Get the range of the chunk'
        strt_row = (ii * batch_size)
        stop_row  = strt_row + batch_size

        'Check for over-flow'
        if stop_row  > full_mat.shape[0]:
            stop_row = full_mat.shape[0]

        dst_chunk = 1. - cosine_similarity(full_mat[strt_row:stop_row, :], full_mat[strt_row:, :])

        distances[strt_row:stop_row, strt_row:] = dst_chunk
        distances[strt_row:, strt_row:stop_row] = dst_chunk.T

    return distances

class segment_set(object):
    def __init__(self, num_elements):
        """
        This class create a set of segemnts. Each segment contains information about
            1) The segment id (seg_id)
            2) The internal distance of segment (int_dist)
            3) The number of elements in the set (num_elem)

        It also tracks the number of unique segments at any time (num_seg)

        :param num_elements (int):
        The nymber of elements in the segmentation at the beginning.
        """

        assert isinstance(num_elements, int) and num_elements > 0, "<num_elements> must be a +ve integer"

        'At initialization each element is a segment by itself, so intialize number of unique segments to number of' \
        'elements'
        self.__num_segments = num_elements
        #self.__num_nodes = num_elements

        'Create a numpy array to hold other aspects of each segment'
        self.__segments = np.empty((num_elements, 3), dtype=float)
        for ii in range(num_elements):
            'Since each segment is seperated give unique id'
            self.__segments[ii, 0] = ii
            'Set internal distance at init to 0, as each segment has only one element'
            self.__segments[ii, 1] = 0
            'Set num_elements at init to 1, as each segment has only one element'
            self.__segments[ii, 2] = 1

    def size(self, seg_id):
        """
        This function returns the number of elements/nodes with a given segment id

        :param seg_id: (int)
        The id of the segment queried

        :return:
        number of elements/nodes with that segment id
        """
        'Check if specified segment exists'
        assert seg_id in self.__segments[:, 0], "Queried segment does not exist"

        'Return number of elements in the segment id'
        return self.__segments[seg_id, 2]


    def num_segs(self):
        """
        Returns the number of unique segments in the segmentation

        :return:
        """
        return self.__num_segments

    def find(self, element_num):
        """
        This function returns the segment id of a specific element

        :param element_num: (int)
        The id of the segment queried
        """

        'Check that the element exists'
        assert isinstance(element_num, int) and element_num >= 0, "<element_num> must be a +ve integer"
        assert element_num < self.__segments.shape[0], f"<element_num> must be in range " \
                                                       f"0-{self.__segments.shape[0]-1}"

        y = int(element_num)
        while y != self.__segments[y, 0]:
            y = int(self.__segments[y, 0])
        self.__segments[element_num, 0] = y

        return y

    def seg_mst(self, segment_id):
        """
        This function returns the MST of a specific segment

        :param segment_id: (int)
        The id of the segment queried
        """

        'Check if the segment id exists'
        assert segment_id in self.__segments[:, 0], f"Queried segment <seg1> = {segment_id} does not exist"

        return self.__segments[segment_id, 1]

    def join(self, seg1, seg2, min_dist):
        """
        This function is used to combine segments.

        :param seg1:
        Segment id of segment-1

        :param seg2:
        Segment id of segment-2

        :param min_dist:
        The minimum distance between the two segments

        :return:
        """

        'Check if specified segments exist'
        assert seg1 in self.__segments[:, 0], f"Queried segment <seg1> = {seg1} does not exist"
        assert seg2 in self.__segments[:, 0], f"Queried segment <seg2> = {seg2} does not exist"

        'Combine into the segment with the lower id'
        if self.__segments[seg1, 0] < self.__segments[seg2, 0]:
            'Update id for second segment'
            self.__segments[seg2, 0] = self.__segments[seg1, 0]

            'Update internal distance and number of elements of first segment'
            self.__segments[seg1, 1] = self.__segments[seg1, 1] + self.__segments[seg2, 1] + min_dist
            self.__segments[seg1, 2] += self.__segments[seg2, 2]
        else:
            'Update id for first segment'
            self.__segments[seg1, 0] = self.__segments[seg2, 0]

            'Update internal distance and number of elements of second segment'
            self.__segments[seg2, 1] += self.__segments[seg1, 1] + min_dist
            self.__segments[seg2, 2] += self.__segments[seg1, 2]

        self.__num_segments -= 1
        return self


    def current_segmentation(self):
        """
        This function return the current segmentation stored in the object.

        :return:
        """

        "Find the segment-id for each node"
        return np.asarray([self.find(ii) for ii in range(self.__segments.shape[0])])






class distance_based_clustering(object):
    """
    This class implements a graph based clustering, wherein it ensures that MST inside each segment is below a fixed
    threshold. Since it can be proved that MST(segment) >= Max_distance(segement) this ensures that the max variance
    inside a segment is below a specific threshold.
    """
    def segment_graph(self, edges, seg_map=None, thresh=0.07):
        """
        This function can be used peform a graph based FH segmentation of data. The function requires the list of edges
        in thw graph which is being segmented.

        :param edges: [numpy.ndarray nSamples X 3]
        A numpy array which holds information on the edges in the graph for image segementation.
            o Column-1: Source Node
            o Column-2: Destination Node
            o Column-3: Edge Weight

        :param seg_map: [segment_map] (Default:None)
        An object of the segment map class. If none is given, one is created which wherein each node is considered a
        segment by itself

        :param thresh: [float > 0] (Default: 0.07)
        The parameter that controls the size of the largest divergence in each segment.

        :return:
        """

        assert isinstance(edges, np.ndarray), "The <edges> varible must be a numpy array"
        assert edges.shape[1] == 3, "The <edges> variable must have 3 columns"
        assert isinstance(thresh, float) and (thresh> 0), "The variable <thresh> must be positive"
        assert all(x>=0 for x in edges[:, -1]), "The final column in <edges> must have the edge weights which must " \
                                                "be >= 0 for all edges"
        'If no segmentation exists create a basic one where each node is a segment by itself'
        if seg_map is None:
            seg_map = segment_set(num_elements=int(edges[:, 1].max() + 1))

        assert isinstance(seg_map, segment_set), "The variable <seg_map> must be an object of the <segment_set> class"


        'Iterate over them one at a time'
        for ii in tqdm(range(edges.shape[0])):
            "Consider the edges one by one"
            src_node = seg_map.find(int(edges[ii, 0]))
            dest_node = seg_map.find(int(edges[ii, 1]))

            "check if the edges connects nodes in different segments"
            if src_node != dest_node:
                "check if MST of combined segment is below the threshold of choice."
                if (seg_map.seg_mst(src_node) + seg_map.seg_mst(dest_node) + edges[ii, 2])< thresh:
                    'If the MST is less than the specified limit MERGE the two nodes'
                    seg_map = seg_map.join(src_node, dest_node, edges[ii, 2])

            #updated_dest_node = seg_map.find(int(edges[ii, 1]))

        'Return the updated segment map'
        return seg_map

    def cleanup_smallsegments(self, edges, seg_map, mpix=15, cleanup_mode="outlier"):
        """
        This function can be used to deal with the small clusters or segments in the final clustering.

        :param edges: [numpy.ndarray nSamples X 3]
        A numpy array which holds information on the edges in the graph for image segementation.
            o Column-1: Source Node
            o Column-2: Destination Node
            o Column-3: Edge Weight

        :param seg_map: [segment_map] (Default:None)
        An object of the segment map class.

        :param mpix=15
        The number of pixels a cluster needs to have below which it is considered a small segment

        :param cleanup_mode: ['str' in ['outlier', 'merge']] (Default: outlier)
        String argument which indicates what is to be done with segments which are very small. If 'outlier' mode is
        chosen, then all the small segments are merged into a super class which are considered the outlier class. If
        mode chosen is 'merge', the small segments are merged with the closest large segments available.

        :return: segmentation [np.ndarray: nSamples]
        A cluster map which shows cluster assignments of each class.
        """

        assert cleanup_mode in SUPPORTED_SMALLSEG_MODES, f"Current implementation only supports the following " \
                                                         f"modes for small segments: {SUPPORTED_SMALLSEG_MODES}"
        assert isinstance(seg_map, segment_set), "The variable <seg_map> must be an object of the <segment_set> class"
        assert isinstance(edges, np.ndarray), "The <edges> varible must be a numpy array"
        assert edges.shape[1] == 3, "The <edges> variable must have 3 columns"

        'If cleanup mode is merge'
        if cleanup_mode == "merge":
            'Iterate over the edges one at a time'
            for ii in range(edges.shape[0]):
                "Consider the edges one by one"
                src_node = seg_map.find(edges[ii, 0])
                dest_node = seg_map.find(edges[ii, 1])


                "check if the edges connects nodes in different segments"
                if (src_node != dest_node) and ((seg_map.size(src_node) < mpix) or (seg_map.size(dest_node) < mpix)):
                    seg_map.join(src_node, dest_node)

            return seg_map.current_segmentation()

        if cleanup_mode == "outlier":
            'Get the intial segmentation'
            segmentation = seg_map.current_segmentation()
            'Get the unique segment ids'
            unq_seg_id, cnts_seg_id = np.unique(segmentation, return_counts=True)
            'Iterate over the segment'
            for ii in unq_seg_id:
                if seg_map.size(ii) < mpix:
                    'Get the current member of segment'
                    current_nodes = np.where(segmentation == ii)[0]
                    'Change label to -1 or outlier/unknown group'
                    segmentation[current_nodes] = -1

            return segmentation

    def segment_data(self, data, num_nn=4, mpix=15, thresh=0.2, metric='cosine', cleanup_mode='outlier'):
        """
        This function can be used to create the graph that will be used to segment the data into appropriate clusters.

        :param [np.ndarray: nSamples X nDims]
        A numpy array, where each row contains the samples which are to be segmented into clusters. Each samples is
        represented by a node in the newly created graph.

        :param num_nn [0 < int < nSamples]
        The number of neighbors to add connections to in the graph created.

        :param mpix=15
        The number of pixels a cluster needs to have below which it is considered a small segment

        :param thresh: [float > 0] (Default: 0.07)
        The parameter that controls the size of the largest divergence in each segment.

        :param metric: ['str' in SUPPORTED_DISTANCES]
        A string that indicates the distance type which is added to the edges

        :param cleanup_mode: ['str' in ['outlier', 'merge']] (Default: outlier)
        String argument which indicates what is to be done with segments which are very small. If 'outlier' mode is
        chosen, then all the small segments are merged into a super class which are considered the outlier class. If
        mode chosen is 'merge', the small segments are merged with the closest large segments available.

        :return:
        """

        assert isinstance(data, np.ndarray), "The <data> variable must be a numpy array"
        assert isinstance(num_nn, int), "The <num_nn> variable must be an integer"
        assert 0 <= num_nn, "The <num_nn> variable must be greater than 0"
        assert num_nn <= data.shape[0], f"<num_nn> must be less than total number of samples. Currently, " \
                                        f"<num_nn>:{num_nn} is greater than number of samples {data.shape[0]}"

        assert isinstance(mpix, int) and (mpix > 0), "<mpix> must be a positive integer"
        assert mpix <= data.shape[0], f"<mpix> must be less than total number of samples. Currently, " \
                                        f"<mpix>:{mpix} is greater than number of samples {data.shape[0]}"

        assert metric in SUPPORTED_DISTANCES, f"Current implementation only supports the following distances:" \
                                                     f" {SUPPORTED_DISTANCES}"
        assert cleanup_mode in SUPPORTED_SMALLSEG_MODES, f"Current implementation only supports the following " \
                                                          f"modes for small segments: {SUPPORTED_SMALLSEG_MODES}"


        'Find the distances between all pairs of points'
        distances = cosine_similarity_chunk(data)

        'Create a variable to hold the edges'
        edges = np.zeros((distances.shape[0] * num_nn, 3))
        ctr = 0

        'collect a list of edges'
        for ii in tqdm(range(distances.shape[0])):
            'Find the num_nn closest point for each samples'
            ind = np.argpartition(distances[ii, :], (num_nn + 1))[:(num_nn + 1)]
            ind = ind[~np.isin(ind, np.asarray([ii]))]

            'Add the edges for the closest neighbors'
            for item in ind:
                edges[ctr, :] = np.asarray([min(ii, item), max(ii, item), distances[ii, item]])
                ctr += 1

        'Eliminate repated rows and sort the edges based on the edge weights'
        edges = np.unique(edges, axis=0)
        edges = edges[edges[:, -1].argsort()]

        'perform the segmentation for this graph using the distance'
        seg_map = self.segment_graph(edges, thresh=thresh)

        'perform appropriate cleanup for the very small segments'
        final_segmentation = self.cleanup_smallsegments(edges, seg_map, cleanup_mode=cleanup_mode)

        return final_segmentation



if __name__ == "__main__":
    'Load the outliers and features from the image'
    glbl_outliers = np.load("test_A425.npy")
    glbl_outliers_feat = np.load("test_A425_feat.npy")

    'Perform a segmentation'
    final_segment = distance_based_clustering().segment_data(glbl_outliers_feat, mpix=5,)

    print('finished')

