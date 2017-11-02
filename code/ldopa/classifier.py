from datamanagement.datasets import dataset, get_dataset
#from datamanagement.utils import batchRandomRotation

from modeldefs import modeldefs
import logging
import itertools
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ReduceLROnPlateau, TensorBoard

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

from keras.models import Model, load_model
from keras.layers import Dense
from keras import backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from sklearn import metrics
from scipy.stats import binom
import sys

from scoring.predictionprobability import pk


def log_choose_4(k):
    if k == 0:
        return 1
    elif k == 1:
        return 4
    elif k == 2:
        return 6
    elif k == 3:
        return 4
    else:
        return 1

def log_choose(n, k):
    x = K.epsilon()
    for i in K.arange(n):
        x += K.log(K.cast_to_floatx(i))
    for i in K.arange(k):
        x -= K.log(K.cast_to_floatx(i))
    for i in K.arange(n-k):
        x -= K.log(K.cast_to_floatx(i))
    return x

def binomial_loss(y_true, y_pred):
    return K.mean(-log_choose_4(y_true) - y_true*K.log(y_pred) - (4-y_true)*K.log(1-y_pred))



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

            if len(dataset.keys()) == 2 and False:
                # special case for meta + timeseries
                # here, either meta or timeseries might be dropped out
                val=np.random.binomial(1, 0.5, 2)
                if np.sum(val) == 0 or np.sum(val) == 2:
                    #keep both dataset
                    continue
                elif val[0] == 1:
                    # set timeseries to zeros
                    Xinput['input_1'] = np.zeros(Xinput['input_1'].shape, dtype="float32")
                else:
                    # set metadata to zeros
                    Xinput['input_2'] = np.zeros(Xinput['input_2'].shape, dtype="float32")

            yinput = dataset['input_1'].labels[
                    indices[ib*batchsize:(ib+1)*batchsize]].copy()

            #yinput = yinput.astype("float32")

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
    n_outputs = {'bra':1, 'dys':1, 'tre':1}
    act_fct = {'bra':'sigmoid', 'dys':'sigmoid', 'tre': 'sigmoid'}
    loss_fct = {'bra':'binary_crossentropy', 'dys': 'binary_crossentropy',
            'tre': binomial_loss}
    # todo evaluation method also depends on task.
    # AUC for 'bra' and 'dys' and perhaps prediction probability (PR) for 'tre'


    def __init__(self, datadict, model_definition, comb, epochs,
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

        self.name = '.'.join(comb)
        self.comb = comb
        self.logger = logging.getLogger(self.name)

        self.outcome_score = comb[0].split('-')[1] #tre, bra or dys
        self.data = datadict
        self.batchsize = 100

        patient_id = datadict['input_1'].patient

        # determine sample weights
        hcdf = pd.DataFrame(patient_id, columns=["patientId"])
        hcvc = hcdf['patientId'].value_counts()
        hcdf["nsamples"] = hcdf['patientId'].map(lambda r: hcvc[r])
        self.sample_weights = 1. / hcdf["nsamples"].values
        self.sample_weights = np.ones((len(datadict['input_1']), ), dtype="float32")


        self.logger.info("Input dimensions:")
        for k in datadict:
            self.logger.info("\t{}: {} x {}".format(k, len(datadict[k]),
                                    datadict[k].shape))

        self.modelfct = model_definition[0]
        self.modelparams = model_definition[1]
        self.epochs = epochs

        self.summary_path = os.path.join(outputdir, "perf_summary")
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.summary_file = os.path.join(self.summary_path, self.name + ".csv")


    def summaryExists(self):
        return os.path.isfile(self.summary_file)


    def defineModel(self):

        inputs, outputs = self.modelfct(self.data, self.modelparams)

        outputs = Dense(self.n_outputs[self.outcome_score],
            activation=self.act_fct[self.outcome_score],
            name="main_output")(outputs)

        model = Model(inputs = inputs, outputs = outputs)

        model.compile(loss=self.loss_fct[self.outcome_score],
                    optimizer='adadelta',
                    metrics=['accuracy'])

        return model


    def fit(self, augment = True):
        self.logger.info("Start training ...")

        bs = self.batchsize

        patient_id = self.data['input_1'].patient
        individuals = np.unique(patient_id)

        predictions = np.zeros((len(patient_id),
                self.n_outputs[self.outcome_score]),dtype="float32")

        checker = np.zeros((len(patient_id)), dtype="bool")

        ## Perform one-hold-out cross-validation
        for i_lo, leave_out in enumerate(individuals):
            # reinit model in every loop

            print '#' * 85
            print 'Running {} excluding subject {} ({}/{})'.format(self.name,
                leave_out, i_lo + 1, len(individuals))
            print '#' * 85

            tmpmodel = self.defineModel()

            train_idxs = np.where(patient_id != leave_out)[0]
            validate_idxs = np.where(patient_id == leave_out)[0]
            self.logger.info(validate_idxs)

            checker[validate_idxs] = ~checker[validate_idxs]

            history = tmpmodel.fit_generator(
                generate_fit_data(self.data, train_idxs, self.sample_weights, bs,
                        augment),
                steps_per_epoch = len(train_idxs)//bs + \
                    (1 if len(train_idxs)%bs > 0 else 0),
                epochs = self.epochs,
                use_multiprocessing = True)

            self.logger.info("Performance after leaving out {} ({} epochs) loss {:1.3f}, acc {:1.3f}".format(
                leave_out, self.epochs, history.history["loss"][-1], history.history["acc"][-1]
            ))

            # predict on held-out subject
            rest = 1 if len(validate_idxs)%self.batchsize > 0 else 0

            predictions[validate_idxs] = tmpmodel.predict_generator(
                generate_predict_data(self.data,
                validate_idxs, self.batchsize, False),
                steps = len(validate_idxs)//self.batchsize + rest)


            del tmpmodel
            K.clear_session()

        self.logger.info("Finished training ...")
        assert np.all(checker), "not all validation indices are present. WTF!"

        vaperf = self.evaluate(predictions)

        self.logger.info("Results written to {}".format(os.path.basename(self.summary_file)))

        # finally with the re-train the model with all subjects
        train_idxs = np.arange(len(patient_id))
        self.dnn = self.defineModel()

        tb_cbl = TensorBoard(log_dir='./logs/{}/'.format(os.path.splitext(os.path.basename(self.summary_file))[0]),
                             histogram_freq=0, batch_size=32, write_graph=False,
                             write_grads=False, write_images=False, embeddings_freq=0,
                             embeddings_layer_names=None, embeddings_metadata=None)


        history = self.dnn.fit_generator(
            generate_fit_data(self.data, train_idxs, self.sample_weights, bs,
                    augment),
            steps_per_epoch = len(train_idxs)//bs + \
                (1 if len(train_idxs)%bs > 0 else 0),
            epochs = self.epochs, use_multiprocessing = True, callbacks = [tb_cbl])


        predictions = self.dnn.predict_generator(
            generate_predict_data(self.data,
            train_idxs, self.batchsize, False),
            steps = len(train_idxs)//self.batchsize +\
             (1 if len(train_idxs)//self.batchsize > 0 else 0))


        trperf = self.evaluate(predictions)

        perf = pd.DataFrame(data=np.concatenate((vaperf.values,
                trperf.values[:,2:]), axis=1))

        perf.to_csv(self.summary_file, header=False, index=False, sep="\t")

        self.dnn.summary()
        self.dnn.summary(print_fn = self.logger.info)

    def saveModel(self):
        if not os.path.exists(outputdir + "/models/"):
            os.mkdir(outputdir + "/models/")
        filename = outputdir + "/models/" + self.name + ".h5"
        self.logger.info("Save model {}".format(filename))
        self.dnn.save(filename)

    def loadModel(self, name):
        filename = outputdir + "/models/" + self.name + ".h5"
        self.logger.info("Load model {}".format(filename))
        self.dnn = load_model(filename)


    def evaluate(self, predicted):

        yinput = self.data['input_1'].labels

        results = []
        if len(self.comb) == 2:
            results += self.comb
        else:
            results += [self.comb[0], '.'.join(self.comb[1:])]

        if len(np.unique(yinput)) == 2:
            # this is to be checked for the binary prediction
            auroc = metrics.roc_auc_score(yinput, predicted)
            prc = metrics.average_precision_score(yinput, predicted)
            acc = metrics.accuracy_score(yinput, predicted.round())
            f1score = metrics.f1_score(yinput, predicted.round())

            results += [auroc, prc, f1score, acc]

        elif len(np.unique(yinput)) > 2:
            # this is used for tremorScore where values from 0 to 4 are possible
            class_true = yinput
            class_predicted = np.argmax(binom.pmf(np.arange(4), 4, predicted))
            class_predicted = np.array([ np.argmax(binom.pmf(np.arange(4), \
                    4, p)) for p in predicted])
            #class_predicted = np.argmax(predicted, axis=1)
            pkscore = pk(class_true, class_predicted)
            acc = metrics.accuracy_score(class_true, class_predicted)

            results += [pkscore, acc]

        else:
            # this is checked if only one class is left
            # then we cannot score
            auroc = np.nan
            prc = np.nan
            acc = np.nan
            f1score = np.nan

            results += [auroc, prc, f1score, acc]

        return pd.DataFrame([results])

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

    parser.add_argument('data', choices = [ k for k in dataset],
            help = "Selection of Datasets")
    parser.add_argument('--name', dest="name", default="", help = "Name-tag")
    parser.add_argument('--epochs', dest="epochs", type=int,
            default=30, help = "Number of epochs")
    parser.add_argument('--noise', dest="noise",
            default=False, action='store_true', help = "Add Gaussian noise to the input")
    parser.add_argument('--rotate', dest="rotate",
            default=False, action='store_true', help = "Data augmentation by rotation")
    parser.add_argument('--flip', dest="flip",
            default=False, action='store_true', help = "Data augmentation by flipping the coord. signs")
    parser.add_argument('--rofl', dest="rofl",
            default=False, action='store_true', help = "Data augmentation by flipping and rotation")

    args = parser.parse_args()

    comb = [args.data, args.model]
    print("{}".format(comb))

    if args.rotate:
        comb += ["rot"]

    da = {}
    for k in dataset[args.data].keys():
        da[k] = get_dataset(dataset[args.data][k])
        if args.noise:
            comb += ["noise"]
            da[k].transformData = da[k].transformDataNoise
        if args.rotate:
            comb += ["rotate"]
            da[k].transformData = da[k].transformDataRotate
        if args.flip:
            comb += ["flip"]
            da[k].transformData = da[k].transformDataFlipSign
        if args.rofl:
            comb += ["rofl"]
            da[k].transformData = da[k].transformDataFlipRotate

    model = Classifier(da,
            modeldefs[args.model], comb=comb,
                        epochs = args.epochs)

    model.fit(args.noise|args.rotate|args.flip|args.rofl)
    model.saveModel()
