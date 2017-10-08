from datamanagement.datasets import dataset
from modeldefs import modeldefs
import logging
import itertools
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ReduceLROnPlateau

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

from keras.models import Model, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from sklearn import metrics
import sys

outputdir = os.getenv('PARKINSON_DREAM_LDOPA_DATA')

def generate_fit_data(dataset, indices, sample_weights, batchsize, augment = True):
    while 1:
        ib = 0
        if len(indices) == 0:
            raise Exception("index list is empty")
        while ib < (len(indices)//batchsize + (1 if len(indices)%batchsize > 0 else 0)):
            Xinput = {}
            for ipname in dataset.keys():
                Xinput[ipname] = dataset[ipname].getData(
                    indices[ib*batchsize:(ib+1)*batchsize], augment).copy()

            yinput = dataset['input_1'].labels[
                    indices[ib*batchsize:(ib+1)*batchsize]].copy()

            sw = sample_weights[indices[ib*batchsize:(ib+1)*batchsize]].copy()

            ib += 1

            if yinput.shape[0] <=0:
                raise Exception("generator produced empty batch")
            yield Xinput, yinput, sw

def generate_predict_data(dataset, indices, batchsize, augment = True):
    while 1:
        ib = 0
        if len(indices) == 0:
            raise Exception("index list is empty")
        while ib < (len(indices)//batchsize + (1 if len(indices)%batchsize > 0 else 0)):
            Xinput = {}
            for ipname in dataset.keys():
                Xinput[ipname] = dataset[ipname].getData(
                    indices[ib*batchsize:(ib+1)*batchsize], augment).copy()

            ib += 1

            yield Xinput

class Classifier(object):
    def __init__(self, datadict, model_definition, name, epochs,
        logs = "model.log", overwrite = False):
        '''
        :input: is a class that contains the input for the prediction
        :model: is a function that defines a keras model for predicting the PD
        :name: used to store the params and logging
        '''
        logdir = os.path.join(outputdir, "logs")
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        logging.basicConfig(filename = "/".join([logdir, logs]),
            level = logging.DEBUG,
            format = '%(asctime)s:%(name)s:%(message)s',
            datefmt = '%m/%d/%Y %I:%M:%S')
        self.logger = logging.getLogger(name)

        self.name = name
        self.outcome_score = name.split('-')[1] #tre, bra or dys
        self.data = datadict
        self.batchsize = 100

        patient_id = datadict['input_1'].patient

        # determine sample weights
        hcdf = pd.DataFrame(patient_id, columns=["patientId"])
        hcvc = hcdf['patientId'].value_counts()
        hcdf["nsamples"] = hcdf['patientId'].map(lambda r: hcvc[r])
        self.sample_weights = 1. / hcdf["nsamples"].values

        self.modelfct = model_definition[0]
        self.modelparams = model_definition[1]
        self.epochs = epochs

        self.summary_path = os.path.join(outputdir, "perf_summary")
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.ind_summary_path = os.path.join(self.summary_path, "individual")
        if not os.path.exists(self.ind_summary_path):
            os.makedirs(self.ind_summary_path)


        self.summary_file = os.path.join(self.summary_path, self.name + ".csv")
        self.ind_summary_file = os.path.join(self.ind_summary_path, self.name + ".csv")

    def summaryExists(self):
        return os.path.isfile(self.ind_summary_file)


    def defineModel(self):

        inputs, outputs = self.modelfct(self.data, self.modelparams)

        if self.outcome_score in ['bra', 'dys']:
            loss_fct = 'binary_crossentropy'
            n_outputs = 1
        elif self.outcome_score == 'tre':
            loss_fct = 'categorical_crossentropy'
            n_outputs = 5
        else:
            raise Exception('Unknown outcome score "{}"'.format(self.outcome_score))

        outputs = Dense(n_outputs, activation='sigmoid', name="main_output")(outputs)

        model = Model(inputs = inputs, outputs = outputs)
        model.compile(loss=loss_fct,
                    optimizer='adadelta',
                    metrics=['accuracy'])

        model.summary()
        model.summary(print_fn = self.logger.info)

        return model

    def fit(self, augment = True):
        self.logger.info("Start training ...")

        bs = self.batchsize

        patient_id = self.data['input_1'].patient
        individuals = np.unique(patient_id)

        perf = pd.DataFrame()

        for i_lo, leave_out in enumerate(individuals):
            # reinit model in every loop

            print '#' * 85
            print 'Running {} excluding subject {} ({}/{})'.format(self.name, leave_out, i_lo + 1, len(individuals))
            print '#' * 85

            self.dnn = self.defineModel()

            train_individuals = np.setdiff1d(individuals, [leave_out])

            train_idxs = np.where(np.in1d(patient_id, train_individuals))
            validate_idxs = np.where(np.in1d(patient_id, leave_out))

            history = self.dnn.fit_generator(
                generate_fit_data(self.data, train_idxs, self.sample_weights, bs,
                        augment),
                steps_per_epoch = len(train_idxs)//bs + \
                    (1 if len(train_idxs)%bs > 0 else 0),
                epochs = self.epochs,
                use_multiprocessing = True)

            self.logger.info("Performance after leaving out {} ({} epochs) loss {:1.3f}, acc {:1.3f}".format(
                leave_out, self.epochs, history.history["loss"][-1], history.history["acc"][-1]
            ))

            perf = perf.append(self.evaluate(train_idxs, validate_idxs, leave_out))

        self.logger.info("Finished training ...")

        perf_summary = perf.drop(["dataset", "model", "subject_left_out"], axis=1)\
            .apply([np.mean, np.std])\
            .transpose()\
            .reset_index()\
            .values.ravel()

        f = open(self.summary_file, 'w')
        f.write(("{}\t{:.3f}\t(+-{:.3f})\t" * 4).format(*perf_summary))
        f.write(self.name.replace('.', '\t'))
        f.write('\n')
        f.close()

        perf.to_csv(self.ind_summary_file,
                    header=True, index=False, sep="\t")

        self.logger.info("Results written to {}".format(os.path.basename(self.summary_file)))

    def saveModel(self):
        raise Exception('Not implemented: Which model should be saved?')
        #if not os.path.exists(outputdir + "/models/"):
        #    os.mkdir(outputdir + "/models/")
        #filename = outputdir + "/models/" + self.name + ".h5"
        #self.logger.info("Save model {}".format(filename))
        #self.dnn.save(filename)

    def loadModel(self, name):
        filename = outputdir + "/models/" + self.name + ".h5"
        self.logger.info("Load model {}".format(filename))
        self.dnn = load_model(filename)


    def evaluate(self, train_idxs, validate_idxs, leave_out):

        yinput = self.data['input_1'].labels

        results = list(self.name.split('.')) + [leave_out]

        for idxs, name in zip([validate_idxs, train_idxs], \
                ['left_out', 'train']):
            y = yinput[idxs]

            rest = 1 if len(idxs)%self.batchsize > 0 else 0

            #scores = self.dnn.predict_generator(generate_predict_data(self.data,
            #    idxs, self.batchsize, False),
            #    steps = len(idxs)//self.batchsize + rest)

            #prc = metrics.average_precision_score(y, scores)
            #acc = metrics.accuracy_score(y, scores.round())
            #results += [acc, prc]

            # use evaluate_generator() for loss and accuracy

            losses = self.dnn.evaluate_generator(
                generate_fit_data(self.data, idxs, self.sample_weights, self.batchsize, False),
                steps=len(idxs) // self.batchsize + rest)

            results += losses

        return pd.DataFrame([results],
                            columns=["dataset", "model", "subject_left_out"] +
                                        ['_'.join(x) for x in itertools.product(['val', 'train'],
                                                           self.dnn.metrics_names)])

    def featurize(self):
        pass

if __name__ == "__main__":

    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(description = \
            'Collection of models for parkinson prediction', formatter_class = \
            RawTextHelpFormatter)
    helpstr = "Models:\n"
    for mkey in modeldefs:
            helpstr += mkey +"\t"+modeldefs[mkey][0].__doc__.format(*modeldefs[mkey][1])+"\n"
    parser.add_argument('model', choices = [ k for k in modeldefs],
            help = "Selection of Models:" + helpstr)
    helpstr = "Datasets:\n"
    for dkey in dataset:
            # for now lets stick with input_1 only
            # TODO: later make this more general
            helpstr += dkey +"\t"+dataset[dkey].__doc__+"\n"
    parser.add_argument('data', choices = [ k for k in dataset],
            help = "Selection of Datasets:" + helpstr)
    parser.add_argument('--name', dest="name", default="", help = "Name-tag")
    parser.add_argument('--epochs', dest="epochs", type=int,
            default=30, help = "Number of epochs")
    parser.add_argument('--augment', dest="augment",
            default=False, action='store_true', help = "Use data augmentation if available")

    args = parser.parse_args()
    name = '.'.join([args.data, args.model])
    print("--augment {}".format(args.augment))
    if args.augment:
        name = '_'.join([name, "rot"])

    da = {}
    for k in dataset[args.data].keys():
        da[k] = dataset[args.data][k]()

    model = Classifier(da,
            modeldefs[args.model], name=name,
                        epochs = args.epochs)

    model.fit(args.augment)
    model.saveModel()
    model.evaluate()
