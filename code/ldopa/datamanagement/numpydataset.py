from __future__ import print_function, division
import os
import joblib
import numpy as np
from ldopa_data import LDopa
import progressbar
from keras.utils.np_utils import to_categorical
import pandas as pd
from .utils import batchRandomRotation


datadir = os.getenv('PARKINSON_DREAM_LDOPA_DATA')
outcome_vars = {'tre': 'tremorScore', 'dys' : 'dyskinesiaScore', 'bra' : 'bradykinesiaScore'}

tremor_tasks = ['drnkg', 'fldng',
                'ramr1', 'raml1', 'ramr2', 'raml2',
                'orgpa',
                'ftnl1', 'ftnr1', 'ftnl2', 'ftnr2',
                'ntblt']

dyskin_tasks = ['ramr1', 'raml1', 'ramr2', 'raml2',
                'ftnl1', 'ftnr1', 'ftnl2', 'ftnr2']

brakin_tasks = ['drnkg', 'fldng', 'orgpa',
               'ramr1', 'raml1', 'ramr2', 'raml2',
               'ftnl1', 'ftnr1', 'ftnl2', 'ftnr2']

subch_tasklist = {'tre': tremor_tasks, 'dys': dyskin_tasks, 'bra': brakin_tasks }



class NumpyDataset(object):
    def __init__(self, outcome, task, reload_ = False):
        """
        outcome = ["tre", "dys", "bra"]
        task = [specific tast, all tasks, or metadata]
        """

        self.outcome = outcome
        self.task = task
        self.reload = reload_

        self.columns = ['x', 'y', 'z']

        if not hasattr(self, 'npcachefile'):
            self.npcachefile = os.path.join(datadir,
                "{}_{}_{}.pkl".format(self.__class__.__name__.lower(), self.outcome, self.task))

        self.load()

    def loadAllTasks(self):
        if not os.path.exists(self.npcachefile) or self.reload:
            ld = LDopa()

            subch = self.outcome
            outcome_var = outcome_vars[subch]

            cdtask = ld.commondescr[(ld.commondescr["task"].isin(subch_tasklist[subch])) & \
                    ~(ld.commondescr[outcome_var].isnull())]
            nrows = cdtask.shape[0]

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

            data = data[keepind]

            labels = cdtask[outcome_var].values


            #if outcome_var == 'tremorScore':
                #labels = pd.DataFrame(to_categorical(labels, num_classes=5), index=labels.index)
                #labels = cdtask[outcome_var].values

            labels = labels[keepind]
            patients = cdtask["patient"].iloc[keepind].values

            joblib.dump((data, labels, keepind, patients), self.npcachefile)

        self.data, self.labels, self.keepind, self.patient = joblib.load(self.npcachefile)

    def loadSpecificTask(self):
        if not os.path.exists(self.npcachefile) or self.reload:

            ld = LDopa()

            outcome_var = outcome_vars[self.outcome]

            cdtask = ld.commondescr[(ld.commondescr["task"] == self.task) & \
                    ~(ld.commondescr[outcome_var].isnull())]
            nrows = cdtask.shape[0]

            if self.task in ['ftnl1', 'ftnl2', 'ftnr1', 'ftnr2',\
                'ramr1','ramr2','raml1','raml2']:
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

            #if outcome_var == 'tremorScore':
                #labels = pd.DataFrame(to_categorical(labels, num_classes=5), index=labels.index)

            labels = labels[keepind]

            annotation = cdtask.iloc[keepind]

            patient = annotation["patient"].values

            joblib.dump((data, labels, keepind, patient), self.npcachefile)

        self.data, self.labels, self.keepind, self.patient = joblib.load(self.npcachefile)


    def loadMeta(self):
        if not os.path.exists(self.npcachefile) or self.reload:

            ld = LDopa()
            subch = self.outcome

            outcome_var = outcome_vars[self.outcome]


            cdtask = ld.commondescr[(ld.commondescr["task"].isin(subch_tasklist[subch])) & \
                    ~(ld.commondescr[outcome_var].isnull())]

            nrows = cdtask.shape[0]

            device_data = cdtask["device"].apply(lambda x: 1 if x=="GENEActiv" else 0)
            #session_data = ld.commondescr["session"].apply(lambda x: 1 if x==1 else 0)
            site_data = cdtask["site"].apply(lambda x: 1 if x=="Boston" else 0)
            tasks = cdtask["task"].values

            tsks = np.unique(tasks)

            df = pd.DataFrame()
            for t in tsks:
                df[t] = cdtask["task"].apply(lambda x: 1 if x==t else 0)
            task_data = df.values
            visit_data = cdtask["visit"].apply(lambda x: 1 if x==1 else 0)
            deviceside_data = cdtask["deviceSide"].apply(lambda x: 1 if x=="Right" else 0)

            keepind = np.ones((nrows), dtype=bool)

            bar = progressbar.ProgressBar()

            for idx in bar(range(nrows)):
                df = ld.loadfile(cdtask['dataFileHandleId'].iloc[idx])

                if df.empty or df["x"].isnull().any().any():
                    keepind[idx] = False

            data = np.concatenate((device_data[:, np.newaxis],
                        site_data[:, np.newaxis], task_data,
                        visit_data[:, np.newaxis],
                        deviceside_data[:, np.newaxis]), axis=1)

            data = data[keepind]

            labels = cdtask[outcome_var].values

            #if outcome_var == 'tremorScore':
                #labels = pd.DataFrame(to_categorical(labels, num_classes=5), index=labels.index)

            labels = labels[keepind]

            annotation = cdtask.iloc[keepind]

            patient = annotation["patient"].values

            joblib.dump((data, labels, keepind, patient), self.npcachefile)

        self.data, self.labels, self.keepind, self.patient = joblib.load(self.npcachefile)

    def load(self):
        if self.task == "all":
            self.loadAllTasks()
        elif self.task == "meta":
            self.loadMeta()
        else:
            self.loadSpecificTask()

    def getData(self, idx = None, transform = False):
        if type(idx) == type(None):
            data = self.data
        else:
            data = self.data[idx]

        if transform:
            return self.transformData(data)
        else:
            return data

    def transformDataNoise(self, data):
        return data + np.random.normal(scale=np.sqrt(0.1), size=data.shape)

    def transformDataRotate(self, data):
        return batchRandomRotation(data)

    def transformDataFlipSign(self, data):
        for t in range(data.shape[0]):
            data[t] = np.matmul(data[t], np.diag(np.random.choice([1,-1], 3)))

        return data

    def transformDataFlipRotate(self, data):
        data = self.transformDataRotate(data)
        data = self.transformDataFlipSign(data)

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


