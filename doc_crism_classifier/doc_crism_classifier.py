"""
FileName:               doc_spectralData.py
Author Name:            Arun M Saranathan
Description:            This code file is used to create Deep Open Classifier (which uses a 1-vs-rest strategy by
                        replacing the penulitimate layer with a bunch of sigmoids rather than a softmax) for CRISM data.
                        The model is trained on highly similar spectra based on my feature extraction and then tested
                        on some CRISM image data

Date Created:           23rd June 2021
Last Modified:          23rd June 2021
"""

import numpy as np
import os
import json
import tensorflow as tf
from scipy.stats import norm as dist_model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical

from hsiUtilities.crism_ganProc_utils import crism_ganProc_utils
from extract_samples.create_train_test_data import create_train_test_data
from spectral_utilities.spectral_utilities import spectral_utilities

class doc_crism_classifier(object):
    def __init__(self, n_classes, input_shape=None, class_mode='param', disrep_model=None, model_name='doc_ganRep',
                 class_names=None, SRC_DIR=None, experiment_id='1', verbose=False,
                 parameters_json=os.path.join('/Volume2/arunFiles/CRISM_minMapping_Ident/python_codeFiles/',
                                              'doc_crism_classifier/parameters.json')):
        """
        The constructor to the open set classifier using an DOC model.

        :param n_classes: (int)
        The number of classes in our model (must be specified by the user).

        :param input_shape: (Tuple: int) (Default: [240, 1])
        This parameter is an integer list describing the shape of the input dataset. Default shape of the spectral
        data is set to 240 X 1, which is the size of NIR spectral data from the CRISM image database in the spectral
        range 1.0 to 2.6 microns.

        :param class_mode: (string in ('none', 'param')) (Default: 'param')
        This decides whether the classification is directly performed on the data or if some param extraction is
        performed

        :param disrep_model: (Keras Model) (Default: None)
        This is a pretrained feature extactor which accepts the imput with the same shape as the input shape

        :param model_name: (str) (Default: "doc_ganRep")
        This string which is the name used to save the models if needed. Default to "doc_ganRep".

        :param class_names: (ndarray: string) (Default: True)
        A list of class names. The length of this list must be equal to the number of classes. By default no class names
        are assumed

        :param SRC_DIR (str) (Default: None)
        This string contains the location where the models are saved if needed

        :param experiment_id: (String) (Default: '1')
        A label under which the models created by an object is stored.

        :param verbose: (Bool) (Default: True)
        This flag decides whether factors like model architecture etc are displayed on the console.

        :param parameters_json (string) (Default: 'parameters.json')
        A JSON file which contains the parameters needed to build classifier mode.
        """
        assert isinstance(n_classes, int) and (n_classes > 0), "The number classes must be a nozero positive integer"
        if input_shape is None:
            input_shape = (240, 1,)
        assert isinstance(input_shape, tuple) and all(isinstance(v, int) for v in input_shape), "This variable must" \
                                                                                                "be a tuple with " \
                                                                                                "integers"
        assert os.path.isfile(parameters_json), "Cannot find the JSON file with parameters"
        assert class_mode in ['none', 'param'], "Unknown classification modes have been designed"
        assert isinstance(disrep_model, tf.keras.Model) or (disrep_model is None), "The representation model must be " \
                                                                                   "a keras model"
        assert isinstance(model_name, str), "The model name must be a string"
        assert isinstance(experiment_id, str), "The experiment ID must be a string"
        if class_names is not None:
            assert isinstance(class_names, np.ndarray) and (len(class_names) == n_classes), \
                "The number of class names must be numpy array with a name for each class"
        assert (SRC_DIR is None) or (os.path.isdir(SRC_DIR)), "The base folder provided does not exist"

        'Update/set the base variables'
        self.__n_classes = n_classes
        self.__input_shape = input_shape
        self.__class_mode = class_mode
        self.class_names = class_names
        if disrep_model is not None:
            self.__disrep_model = disrep_model
        else:
            self.__class_mode = 'none'

        '--------------------------------------------------------------------------------------------------------------'
        'Get the Model Parameters from the JSON file'
        with open(parameters_json, 'r') as file:
            parameters = json.load(file)
            self.__parameters = parameters

        'Create the classification model as needed'
        self.doc_classModel = self.create_baseDocModel(model_type=parameters["model_type"],
                                                       class_mode=parameters["class_mode"],
                                                       **parameters["model_info"])
        self.doc_classModel.compile(optimizer=tf.keras.optimizers.Adam(lr=parameters["starting_lr"]),
                                    loss=tf.keras.losses.BinaryCrossentropy(),
                                    metrics=[tf.keras.metrics.BinaryAccuracy()])

        if verbose:
            self.doc_classModel.summary()

        '--------------------------------------------------------------------------------------------------------------'
        'initialize the source directories where the data is to be saved'
        if SRC_DIR is None:
            SRC_DIR = os.getcwd()

        SRC_DIR = os.path.join(SRC_DIR, ('{}-{}'.format(model_name, experiment_id)))

        'Define the session variable'
        self.tmp_dir_path = os.path.join(SRC_DIR, 'tmp')

    def create_baseDocModel(self, class_mode, model_type, n_nodes, dropout=0.2, batch_norm=None):
        """
        This function is used to create a classification model for the DOC. If the class mode is 'param' the
        classification is built on top of a param extractor else the classifier is built directly on the data

        :param class_Mode: [string in ["none", "param"]]
        This decides whether the classification is directly performed on the data or if some param extraction is
        performed.

        :param model_type: [string in ["Dense", "Conv"]]
        This argument controls the model type. It can either be a dense network or a convolutional network

        :param n_nodes:
        This argument controls both the depth of the network (equal to length if n_nodes) and the breadth of the network
        each value in this array is the number of nodes at a specific level.

        :param dropout: [0 <= float <= 1] (Default:0.2)
        The amount of dropout that is permitted for the model

        :return: A keras model for classifiying 1-vs-rest
        """
        assert len(n_nodes) > 0, "The model must contain atleast one layer"
        if dropout is not None:
            assert (dropout >= 0) and (dropout <= 1), "Dropout must between 0 & 1"
        if batch_norm is not None:
            assert (batch_norm >= 0) and (batch_norm <= 1), "Momentum of the batch normalization must between 0 & 1"

        if model_type == "Dense":
            return self.create_denseClassModel(class_mode=class_mode, n_nodes=n_nodes, dropout=dropout,
                                               batch_norm=batch_norm)

        if model_type == "Conv":
            return self.create_convClassModel(class_mode=class_mode, n_nodes=n_nodes, dropout=dropout,
                                              batch_norm=batch_norm)

    def create_denseClassModel(self, class_mode, n_nodes, dropout=0.2, batch_norm=None):
        """
        This function can be used to create a dense classifier network

        :param class_mode: [string in ["none", "param"]]
        This decides whether the classification is directly performed on the data or if some param extraction is
        performed.

        :param n_nodes:
        This argument controls both the depth of the network (equal to length if n_nodes) and the breadth of the network
        each value in this array is the number of nodes at a specific level.

        :param dropout: [0 <= float <= 1] (Default:0.2)
        The amount of dropout that is permitted for the model

        :return: A keras model for classifiying 1-vs-rest
        """

        'Create an input variable'
        input = tf.keras.Input(shape=self.__input_shape)

        'If process in feature space'
        if self.__class_mode == "param":
            assert self.__disrep_model is not None, "A feature extractor has to be provided."

            'Make sure that the layers of the feature extractor are not trainable'
            for layer in self.__disrep_model.layers[:]:
                layer.trainable = False

            x = self.__disrep_model(input)

        else:
            x = input

        'Now apply all the models which we require'
        for nodes in n_nodes:
            'First apply the dense layer'
            x = tf.keras.layers.Dense(nodes)(x)
            'Apply an activation'
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            'Apply dropout if needed'
            if dropout != None:
                x = tf.keras.layers.Dropout(dropout)(x)
            'Apply batch normalization if needed'
            if batch_norm != None:
                x = tf.keras.layers.BatchNormalization(momentum=batch_norm)(x)

        'Now get the output for this layer'
        x = tf.keras.layers.Dense(self.__n_classes)(x)
        output = tf.keras.layers.Activation("sigmoid")(x)
        'Create the full model'
        model = tf.keras.models.Model(inputs=input, outputs=output)

        return model

    def create_convClassModel(self, class_mode, n_nodes, dropout=0.2, batch_norm=None):
        """
        This function can be used to create a convolutional classifier network

        :param class_mode: [string in ["none", "param"]]
        This decides whether the classification is directly performed on the data or if some param extraction is
        performed.

        :param n_nodes:
        This argument controls both the depth of the network (equal to length if n_nodes) and the breadth of the network
        each value in this array is the number of nodes at a specific level.

        :param dropout: [0 <= float <= 1] (Default:0.2)
        The amount of dropout that is permitted for the model

        :return: A keras model for classifiying 1-vs-rest
        """

        'Create an input variable'
        input = tf.keras.Input(shape=(self.__inputShape,))

        'If process in feature space'
        if self.__class_mode == "param":
            assert self.__disrep_model != None, "A feature extractor has to be provided."

            'Make sure that the layers of the feature extractor are not trainable'
            for layer in self.__disrep_model.layers[:]:
                layer.trainable = False

            x = self.__disrep_model(input)

        else:
            x = input

        'Now apply all the models which we require'
        for nodes in n_nodes:
            'First apply the dense layer'
            x = tf.keras.layers.Conv1D(filters=nodes, kernel_size=(11,), strides=2, padding="same")(x)
            'Apply an activation'
            x = tf.keras.layers.LeakyReLU(0.2)(x)
            'Apply dropout if needed'
            if dropout != None:
                x = tf.keras.layers.Dropout(dropout)(x)
            'Apply batch normalization if needed'
            if batch_norm != None:
                x = tf.keras.layers.BatchNormalization(momentum=batch_norm)(x)

        'Now get the output for this layer'
        x = tf.keras.layers.Conv1D(filters=self.__n_classes, kernel_size=(11,), strides=2, padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        output = tf.keras.layers.Activation("sigmoid")(x)
        'Combine these two to make the model'
        model = tf.keras.models.Model(inputs=input, outputs=output)

        return model

    def fit(self, X_train, y_train, val_data=None, save_flag=True, early_stopping=True):
        """
        Train the Deep open class classifier with a one-vs-rest scheme.

        :param X_train: [ndarray: float {nSmaples X inputSize}]
        The 1-D spectral training data.

        :param y_train: [ndarray: float {nSmaples X n_classes}]
        The one hot training labels associated with these spectral data

        :param val_data: [(ndarray, ndarray)] (Default: None)
        A set variable which contains both the validation data and the validation labels

        :param save_flag: [Boolean] (Default: True)
        Boolean variabe which decides whether the best performing model is saved by the model or not.

        :param early_stopping: [Boolean] (Default: True)
        Boolean variabe which decides whether early stopping is used in the model or not.

        :return:
        """

        'Create an object to hold required call backs'
        cBacks = []

        if save_flag:
            'If needed save the model and its architecture'
            if not os.path.exists(os.path.join(os.path.split(self.tmp_dir_path)[0], 'doc_model')):
                os.makedirs(os.path.join(os.path.split(self.tmp_dir_path)[0], 'doc_model'))

            'If the architecture has not been saved - save it'
            modelArch = os.path.join(os.path.split(self.tmp_dir_path)[0], 'doc_model', 'modelArch.json')
            model_json = self.doc_classModel.to_json()
            with open(modelArch, "w") as json_file:
                json_file.write(model_json)

            'Create a call back to save the best performing model'
            bestModel_path = os.path.join(os.path.split(self.tmp_dir_path)[0], 'doc_model', 'bestModel.h5')
            checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=bestModel_path, verbose=True,
                                                              save_best_only=True)
            cBacks.append(checkpointer)

        if early_stopping:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            cBacks.append(early_stop)

        'Train the model'
        self.doc_classModel.fit(X_train, y_train, validation_data=val_data, callbacks=cBacks,
                                **self.__parameters["fit_params"])

        'Get the labels predicted for the training data'
        y_pred = self.doc_classModel.predict(X_train)
        params_gauss = []
        classNum_y = np.argmax(y_train, axis=1)

        'Calculate the parameters by fitting a gaussian to the data'
        for ii in range(self.__n_classes):
            pos_mu, pos_std = self.fit_gauss(y_pred[classNum_y == ii, ii])
            params_gauss.append([pos_mu, pos_std])

        if save_flag:
            paramsLoc = os.path.join(os.path.split(self.tmp_dir_path)[0], 'doc_model', 'classGaussParams.npy')
            with open(paramsLoc, 'wb') as f:
                np.save(f, params_gauss)

            if self.class_names is not None:
                class_name_loc = os.path.join(os.path.split(self.tmp_dir_path)[0], 'doc_model', 'class_names.npy')
                with open(class_name_loc, 'wb') as f:
                    np.save(f, self.class_names)

        self.class_params_gauss = params_gauss

    def predict_test(self, X, scale=1.):
        """
        This function can be used to predict the whether the provided data belongs to one of the known classes from the
        training of if it belongs to an unknown class

        :param X: [ndarray: float {nSmaples X inputSize}]
        The 1-D spectral training for prediction.

        :param scale [float] (Default: 2.0)
        This parameter controls the size of the openspace for a specific class.

        :return:  [ndarray: float {nSmaples}]
        The output is the class prediction for each sample. The reject option are the samples labeled with nClass.
        """
        'Get the predictions from the model'
        y_pred = self.doc_classModel.predict(X)
        test_y_pred = []

        'Check each prediction to see if the sample is an outlier to the closest class'
        for p in y_pred:
            max_class, max_value = np.argmax(p), np.max(p)
            'Calcuclate the threshold based on the variance of the half normal distribution'
            # if max_class != 4:
            threshold = min(0.99, 1. - scale * self.class_params_gauss[max_class][1])
            # else:
            #    threshold = min(0.95, 1. - scale * self.class_params_gauss[max_class][1])
            if max_value > threshold:
                test_y_pred.append(max_class)
            else:
                test_y_pred.append(-1)

        return test_y_pred

    def predict_crism(self, X, scale=1., thresh_level=0.95):
        """
        This function can be used to predict the whether the provided data belongs to one of the known classes from the
        training of if it belongs to an unknown class -- the

        :param X: [ndarray: float {nSmaples X inputSize}]
        The 1-D spectral data for prediction.

        :param scale [float] (Default: 2.0)
        This parameter controls the size of the openspace for a specific class.

        :param thresh_level [float](Default: 0.95)
        The minimum value be

        :return:  [ndarray: float {nSmaples X nClasses}]
        The output is vector for each class which show the similarity of each class. The vector is modified for each
        vector such that if the maximal activation is outside the expected distribution of the class then the maximal
        value is replaced by 2 to indicate that these pixels are outliers.
        """

        'Get the predictions from the model'
        y_pred = self.doc_classModel.predict(X)
        test_y_pred = np.zeros(y_pred.shape)

        'Check each prediction to see if the sample is an outlier to the closest class'
        for ii, p in enumerate(y_pred):
            'Identify the class that is closest to the sample'
            max_class, max_value = np.argmax(p), np.max(p)
            'Calcuclate the threshold based on the variance of the half normal distribution'
            threshold = min(0.9, 1. - scale * self.class_params_gauss[max_class][1])
            'Based on threshold decide if a samples is an outlier or a detection of the specific class'
            if max_value > threshold:
                test_y_pred[ii, max_class] = max_value
            else:
                test_y_pred[ii, max_class] = 2.

        return test_y_pred

    def load_preTrained_model(self, model_loc, model_name=None, class_params=None, class_names=None):
        """
        This function can be used train the pre-trained DOC model.

        :param model_loc: [string]
        The folder which contains the saved model.

        :param model_name: [string] (Default: None)
        The name of the file which contains the model architecture and weights. If none is provided the function assumes
        that the file is named 'bestModel.h5'

        :param class_params: [string] (Default: None)
        The name of the file which contains the calculated params for each of the classes found for the specific
        model. If none the function assumes the filename is ''

        :return:
        No return, updates the DOC object to contain the saved model.
        """

        'If no name is provided assume default name for the model'
        if model_name == None:
            model_name = 'bestModel.h5'

        'Combine location and name to get the full model address'
        model_name = os.path.join(model_loc, model_name)
        'load this model onto the object'
        self.doc_classModel = tf.keras.models.load_model(model_name)

        'If no file with gaussian params is provided assume the default name'
        if class_params is None:
            class_params = 'classGaussParams.npy'

        class_params = os.path.join(model_loc, class_params)
        'load this data'
        self.class_params_gauss = np.load(class_params)

        'If no class names are given check for default class names'
        if class_names is None:
            class_names = 'class_names.npy'
        class_names = os.path.join(model_loc, class_names)
        if os.path.isfile(class_names):
            self.class_names = np.load(class_names, allow_pickle=True)
        else:
            self.class_names = None

    def fit_gauss(self, prob_pos_X, reflect=True):
        """
        This function can be used to fit a gaussian model to a set of 1d samples. If reflect option is set to true it
        assumes that the data is sampled from the half distribution of the data.

        :param prob_pos_X: [ndarray: nSamples, float]
        This is the set of 1d samples that is the the input to the data.

        :param reflect: [Boolean] (Default: True)
        Assumes that the data is sampled from a half gaussian distribution

        :return:
        Returns the mean and standard deviation of the set of samples.
        """

        if reflect:
            prob_pos = [p for p in prob_pos_X] + [2 - p for p in prob_pos_X]
        else:
            prob_pos = [p for p in prob_pos_X]

        'Fit the gaussian to the data'
        pos_mu, pos_std = dist_model.fit(prob_pos)

        return pos_mu, pos_std

    def get_base_features(self, X, scl_lvl = 0.2):
        """
        This function can be used to predict the whether the provided data belongs to one of the known classes from the
        training of if it belongs to an unknown class -- the

        :param X: [ndarray: float {nSmaples X inputSize}]
        The 1-D spectral data for prediction.

        :param scl_lvl: [0<= float <= 1](Default: None)
        The level to which the spectra are scaled before doing feature extraction

        :return:  [ndarray: float {nSmaples X nDim}]
        The output for each sample is the activation vector generated by the base feature extraction model of the DOC
        """

        assert self.__disrep_model is not None, "A base feature extractor has to be defined to use the" \
                                                " function 'get_base_features'"
        assert isinstance(X, np.ndarray) and (len(X.shape)==2), "The input variable must be 2D numpy matrix"
        assert X.shape[1] == self.__input_shape[0], "The size of the provided vectors does not match feature " \
                                                    "extractor input size"
        if scl_lvl is not None: assert (0 <= scl_lvl <= 1), "The scaling level must be between 0 & 1"

        if scl_lvl is not None:
            X = spectral_utilities().scale_spectra_cr(X, scale_lvl=scl_lvl)

        'Calculate and return features'
        return self.__disrep_model.predict(np.expand_dims(X, axis=2))

if __name__ == "__main__":
    'LOAD TRAINING AND TEST DATA'
    store_loc = '/Volume2/arunFiles/CRISM_minMapping_Ident/python_codeFiles/extract_samples/data_stores/mica_images.h5'
    (x_train, x_test, y_train, y_test) = create_train_test_data(store_loc).get_train_test('labels', n_samp= 4000,
                                                                                          scl_lvl=0.2)
    'Save the train and test data'
    np.savez('data.npz', name1=x_train, name2=x_test, name3=y_train, name4=y_test)

    #data = np.load('data.npz')
    #x_train, x_test = data['name1'], data['name2']l
    #y_train, y_test = data['name3'], data['name4']

    'Estimate the number of classes'
    n_classes = max(max(y_train), max(y_test)) + 1

    'Get the names of the classes'
    class_names = create_train_test_data(store_loc).get_classnames("endmembers")

    'Create the feature extractor of interest'
    dis_rep = crism_ganProc_utils().create_rep_model()

    'Create an object for the Deep Open Classification (DOC)'
    obj1 = doc_crism_classifier(int(n_classes), disrep_model=dis_rep, model_name='doc_ganRep_crismData',
                                class_names=class_names)
    'Now train the model'
    obj1.fit(x_train, to_categorical(y_train), val_data=(x_test, to_categorical(y_test)))

    'Load a pretrained model'
    #obj1.load_preTrained_model(model_loc=os.path.join('/Volume2/arunFiles/CRISM_minMapping_Ident/python_codeFiles',
    #                                                 'doc_crism_classifier/doc_ganRep_crismData-1/doc_model'))

    """'Also get data from the open-set'
    #x_openset, y_openset= create_train_test_data(store_loc).get_openset('labels', n_samp=15000, scl_lvl=0.2)
    #np.savez('data_open.npz', name1=x_openset, name2=y_openset)

    'Combine the training and open-set data'
    data_open = np.load('data_open.npz')
    x_openset, y_openset = data_open['name1'], data_open['name2']
    x_full, y_full = np.vstack((x_test, x_openset)), np.hstack((y_test, y_openset))

    'Get the prediction of the DOC on the testdata'
    y_testpred = np.asarray(obj1.predict_test(x_full, scale=2.5))

    conf_mat = confusion_matrix(y_full, y_testpred)

    print(conf_mat)
    print(np.sum(np.diag(conf_mat)))
    print('finished')"""