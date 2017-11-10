from __future__ import print_function, division
import os
import joblib
import cv2
import numpy as np
from ldopa_data import LDopa
import progressbar
from keras.utils.np_utils import to_categorical
import pandas as pd
from utils import batchRandomRotation, batchRandomRotationFull
import itertools


datadir = os.getenv('PARKINSON_DREAM_LDOPA_DATA')
outcome_vars = {'tre': 'tremorScore', 'dys' : 'dyskinesiaScore', 'bra' : 'bradykinesiaScore'}


tremor_tasks = ['drnkg', 'fldng',
                #'ramr1', 'raml1',
                'orgpa',
                #'ftnl1', 'ftnr1',
                'ntblt',
                'ramr', 'raml', 'ftnl', 'ftnr',
                'ram', 'ftn']

dyskin_tasks = [#'ramr1', 'raml1', 'ftnl1', 'ftnr1',
                'ramr', 'raml','ftnl', 'ftnr',
                'ram', 'ftn']

brakin_tasks = ['drnkg', 'fldng', 'orgpa',
                #'ramr1', 'raml1', 'ftnl1', 'ftnr1',
                'ramr', 'raml','ftnl', 'ftnr',
                'ram', 'ftn']

subch_tasklist = {'tre': tremor_tasks, 'dys': dyskin_tasks, 'bra': brakin_tasks }



class NumpyDataset(object):
    def __init__(self, outcome, task, reload_ = False, mode="training"):
        """
        outcome = ["tre", "dys", "bra"]
        task = [specific tast, all tasks, or metadata]
        """

        self.outcome = outcome
        self.task = task
        self.reload = reload_
        self.mode = mode

        self.columns = ['x', 'y', 'z']

        if not hasattr(self, 'npcachefile'):
            self.npcachefile = os.path.join(datadir,
                "{}_{}_{}_{}.pkl".format(self.__class__.__name__.lower(),
                    self.outcome, self.task, self.mode))

        self.load()

    def loadAllTestTasks(self):
        if not os.path.exists(self.npcachefile) or self.reload:

            testdata = []
            testfileid= []

            subch = self.outcome
            outcome_var = outcome_vars[subch]

            subTpl = os.path.join('templates', "{}SubmissionTemplate.csv".format(outcome_var[:-5]))
            subTpl = pd.read_csv(subTpl)

            for mode in ["training", "test"]:
                cdtask, ld = self.getCdTasks(mode)

                if subch == 'bra':
                    # otherwise there will be different number of tasks in training in test
                    # and combining the metadata from training and test will fail
                    cdtask = cdtask[cdtask['dataFileHandleId'].isin(subTpl['dataFileHandleId'])]

                nrows = cdtask.shape[0]
                print("all - nrows={}".format(nrows))

                data, keepind = self.getTimeseriesData(cdtask, ld)

                #labels = cdtask[outcome_var].values

                #if outcome_var == 'tremorScore':
                    #if self.mode == "training":
                        #labels = to_categorical(labels, num_classes=5)
                    #else:
                        #labels = np.zeros((cdtask.shape[0], 5))

                filehandles = cdtask.dataFileHandleId.values
                testdata.append(data)
                testfileid.append(filehandles)

            testdata = np.concatenate(testdata, axis=0)
            testfileid = np.concatenate(testfileid, axis=0)
            joblib.dump((testdata, testfileid), self.npcachefile)

        self.data, self.filehandles = joblib.load(self.npcachefile)

    def getTimeseriesData(self, cdtask, ld):
        nrows = cdtask.shape[0]
        print("all - nrows={}".format(nrows))

        # lets just always take 2000 timepoints
        ndatapoints = 2000

        data = np.zeros((nrows, ndatapoints, 3), dtype="float32")
        keepind = np.ones((nrows), dtype=bool)

        bar = progressbar.ProgressBar()

        for idx in bar(range(nrows)):
            df = ld.loadfile(cdtask['dataFileHandleId'].iloc[idx])

            if df.empty or df[self.columns].isnull().any().any():
                keepind[idx] = False
                continue

            df = self.getValues(df)
            data[idx, :min(df.shape[0], ndatapoints), :] = df[:min(df.shape[0], ndatapoints), :]

        return data, keepind

    def loadAllTasks(self):
        if not os.path.exists(self.npcachefile) or self.reload:
            subch = self.outcome
            outcome_var = outcome_vars[subch]

            cdtask, ld = self.getCdTasks("training")

            data, keepind = self.getTimeseriesData(cdtask, ld)

            data = data[keepind]
            print("all (after keepind) - nrows={}".format(data.shape[0]))

            labels = cdtask[outcome_var].values

            if outcome_var == 'tremorScore':
                labels = to_categorical(labels, num_classes=5)

            labels = labels[keepind]
            patients = cdtask["patient"].iloc[keepind].values

            joblib.dump((data, labels, keepind, patients), self.npcachefile)

        self.data, self.labels, self.keepind, self.patient = joblib.load(self.npcachefile)

    def loadSpecificTask(self):
        if not os.path.exists(self.npcachefile) or self.reload:

            ld = LDopa()

            outcome_var = outcome_vars[self.outcome]

            if self.task in ['ramr', 'raml', 'ftnl', 'ftnr']:
                task_names = [''.join(x) for x in itertools.product([self.task], ['1', '2'])]
            elif self.task in ['ram', 'ftn']:
                task_names = [''.join(x) for x in itertools.product([self.task], ['l', 'r'], ['1', '2'])]
            else:
                task_names = [self.task]

            cdtask = ld.commondescr[(ld.commondescr["task"].isin(task_names)) &
                                    ~(ld.commondescr[outcome_var].isnull())]

            nrows = cdtask.shape[0]

            if self.task in ['ftnl1', 'ftnl2', 'ftnr1', 'ftnr2',
                'ramr1','ramr2','raml1','raml2',
                'ramr', 'raml', 'ftnl', 'ftnr',
                'ram', 'ftn']:
                ndatapoints = 1000
            else:
                ndatapoints = 2000

            data = np.zeros((nrows, ndatapoints, 3), dtype="float32")
            keepind = np.ones((nrows), dtype=bool)

            bar = progressbar.ProgressBar()

            for idx in bar(range(nrows)):
                df = ld.loadfile(cdtask['dataFileHandleId'].iloc[idx])

                if df.empty or df[self.columns].isnull().any().any():
                    keepind[idx] = False
                    continue

                df = self.getValues(df)
                data[idx, :min(df.shape[0], ndatapoints), :] = df[:min(df.shape[0], ndatapoints), :]

            data = data[keepind]

            labels = cdtask[outcome_var].values

            if outcome_var == 'tremorScore':
                labels = to_categorical(labels, num_classes=5)
                #labels = labels.values

            labels = labels[keepind]

            annotation = cdtask.iloc[keepind]

            patient = annotation["patient"].values

            joblib.dump((data, labels, keepind, patient), self.npcachefile)

        self.data, self.labels, self.keepind, self.patient = joblib.load(self.npcachefile)

    def getCdTasks(self, mode):
        ld = LDopa(mode=mode)

        subch = self.outcome
        outcome_var = outcome_vars[subch]

        if mode == "training":
            cdtask = ld.commondescr[~(ld.commondescr[outcome_var].isnull())]
        else:
            cdtask = ld.commondescr[ld.commondescr[outcome_var] == "Score"]
        return cdtask, ld


    def loadMetaTest(self):
        if not os.path.exists(self.npcachefile) or self.reload:
            testdata = []
            testfileid= []

            subch = self.outcome
            outcome_var = outcome_vars[subch]

            subTpl = os.path.join('templates', "{}SubmissionTemplate.csv".format(outcome_var[:-5]))
            subTpl = pd.read_csv(subTpl)

            for mode in ["training", "test"]:

                cdtask, ld = self.getCdTasks(mode)

                if subch == 'bra':
                    # otherwise there will be different number of tasks in training in test
                    # and combining the metadata from training and test will fail
                    cdtask = cdtask[cdtask['dataFileHandleId'].isin(subTpl['dataFileHandleId'])]

                data = self.getMetaData(cdtask)
                #data, keepind = getTimeseriesData(cdtask, ld)

                #labels = cdtask[outcome_var].values

                #if outcome_var == 'tremorScore':
                    #if self.mode == "training":
                        #labels = to_categorical(labels, num_classes=5)
                    #else:
                        #labels = np.zeros((cdtask.shape[0], 5))

                filehandles = cdtask.dataFileHandleId.values
                testdata.append(data)
                testfileid.append(filehandles)

                #joblib.dump((data, filehandles), self.npcachefile)
            testdata = np.concatenate(testdata, axis=0)
            testfileid = np.concatenate(testfileid, axis=0)
            joblib.dump((testdata, testfileid), self.npcachefile)

        self.data, self.filehandles = joblib.load(self.npcachefile)

    def getMetaData(self, cdtask):
        device_data = cdtask["device"].apply(lambda x: 1 if x=="GENEActiv" else 0)
        #session_data = ld.commondescr["session"].apply(lambda x: 1 if x==1 else 0)
        site_data = cdtask["site"].apply(lambda x: 1 if x=="Boston" else 0)
        tasks = cdtask["task"].values

        tsks = np.unique(tasks)
        print(tsks)

        df = pd.DataFrame()
        for t in tsks:
            df[t] = cdtask["task"].apply(lambda x: 1 if x==t else 0)
        task_data = df.values
        visit_data = cdtask["visit"].apply(lambda x: 1 if x==1 else 0)
        deviceside_data = cdtask["deviceSide"].apply(lambda x: 1 if x=="Right" else 0)

        data = np.concatenate((device_data[:, np.newaxis],
                    site_data[:, np.newaxis], task_data,
                    visit_data[:, np.newaxis],
                    deviceside_data[:, np.newaxis]), axis=1)

        return data



    def loadMeta(self):
        if not os.path.exists(self.npcachefile) or self.reload:
            subch = self.outcome
            outcome_var = outcome_vars[subch]

            cdtask, ld = self.getCdTasks("training")


            _, keepind = self.getTimeseriesData(cdtask, ld)

            data = self.getMetaData(cdtask)

            data = data[keepind]
            print("meta (after keepind) - nrows={}".format(data.shape))

            labels = cdtask[outcome_var].values

            if outcome_var == 'tremorScore':
                labels = to_categorical(labels, num_classes=5)

            labels = labels[keepind]

            annotation = cdtask.iloc[keepind]

            patient = annotation["patient"].values

            joblib.dump((data, labels, keepind, patient), self.npcachefile)

        self.data, self.labels, self.keepind, self.patient = joblib.load(self.npcachefile)

    def load(self):

        if self.task == "all":
            if self.mode == "training":
                self.loadAllTasks()
            else:
                self.loadAllTestTasks()

        elif self.task == "meta":
            if self.mode == "training":
                self.loadMeta()
            else:
                self.loadMetaTest()
        else:
            self.loadSpecificTask()

    def getData(self, idx = None, transform = False):
        if type(idx) == type(None):
            data = self.data
        else:
            data = self.data[idx]

        if hasattr(self, "transformData"):
            return self.transformData(data)
        else:
            return data

    def transformDataNoise(self, data):
        return data + np.random.normal(scale=np.sqrt(0.1), size=data.shape)

    def transformDataRotate(self, data):
        return batchRandomRotation(data)

    def transformDataRotateFull(self, data):
        return batchRandomRotationFull(data)

    def transformDataFlipSign(self, data):
        for t in range(data.shape[0]):
            data[t] = np.matmul(data[t], np.diag(np.random.choice([1,-1], 3)))

        return data

    def transformDataScaleTimeaxis(self, data):
        for t in range(data.shape[0]):
            rdata = cv2.resize(data[t], (3, int(data[t].shape[0]*np.random.uniform(0.8,1.2))))
            if rdata.shape[0] <= data[t].shape[0]:
                data[t,:rdata.shape[0],:] = rdata
            else:
                data[t] = rdata[data.shape[0], :]
        return data

    def transformDataScaleMagnitude(self, data):
        for t in range(data.shape[0]):
            data[t] =  data[t] * np.random.uniform(0.8,1.2)
        return data


    def transformDataFlipRotate(self, data):
        data = self.transformDataRotate(data)
        data = self.transformDataFlipSign(data)

        return data

    def transformDataSwapDims(self, data):
        e = np.eye(3)
        for t in range(data.shape[0]):
            data[t] = np.matmul(data[t],
                e[np.random.choice(3, size=3, replace=False)])
        return data

    def transformDataReverse(self, data):
        idx_flip = np.where(np.random.randint(0, 2, data.shape[0]))
        data[idx_flip] = np.flip(data[idx_flip], axis=1)

        return data

    def transformDataPermute(self, data):
        n_permute = 50

        for t in range(data.shape[0]):
            ind = data[t].reshape(-1, 3, n_permute)
            np.random.shuffle(ind)
            data[t] = ind.reshape(data[t].shape)

        return data

    def transformDataAll(self, data):
        data = self.transformDataRotate(data)
        data = self.transformDataFlipSign(data)
        data = self.transformDataScaleTimeaxis(data)
        data = self.transformDataScaleMagnitude(data)
   #     data = self.transformDataSwapDims(data)

        return data

    def transformDataAll_v2(self, data):
        data = self.transformDataRotateFull(data)
        data = self.transformDataScaleTimeaxis(data)
        data = self.transformDataScaleMagnitude(data)
        data = self.transformDataPermute(data)
        #     data = self.transformDataSwapDims(data)

        return data

    def transformDataAll_v3(self, data):
        data = self.transformDataRotate(data)
        data = self.transformDataFlipSign(data)
        data = self.transformDataScaleTimeaxis(data)
        data = self.transformDataScaleMagnitude(data)
        data = self.transformDataPermute(data)
   #     data = self.transformDataSwapDims(data)

        return data

    @property
    def patient(self):
        return self.pat

    @patient.setter
    def patient(self, pat):
        self.pat = pat


    @property
    def labels(self):
        return self.lab

    @labels.setter
    def labels(self, lab):
        self.lab = lab

    @property
    def shape(self):
        return self.data.shape[1:]

    def __len__(self):
        return self.data.shape[0]


