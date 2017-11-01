from __future__ import print_function, division
import os
import joblib
import numpy as np
from ldopa_data import LDopa
import progressbar
from keras.utils.np_utils import to_categorical
import pandas as pd
from .utils import batchRandomRotation
import itertools


datadir = os.getenv('PARKINSON_DREAM_LDOPA_DATA')
outcome_vars = {'tre': 'tremorScore', 'dys' : 'dyskinesiaScore', 'bra' : 'bradykinesiaScore'}

class NumpyDataset(object):
    def __init__(self, outcome, task, reload_ = False):

        self.outcome = outcome
        self.task = task
        self.reload = reload_

        self.columns = ['x', 'y', 'z']

        if not hasattr(self, 'npcachefile'):
            self.npcachefile = os.path.join(datadir,
                "{}_{}_{}.pkl".format(self.__class__.__name__.lower(), self.outcome, self.task))

        self.load()

    def load(self):
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

            labels = cdtask[outcome_var]

            if outcome_var == 'tremorScore':
                labels = pd.DataFrame(to_categorical(labels, num_classes=5), index=labels.index)

            labels = labels[keepind]

            joblib.dump((data, labels, keepind), self.npcachefile)

        self.data, self.labels, self.keepind = joblib.load(self.npcachefile)

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
        ld = LDopa()
        outcome_var = outcome_vars[self.outcome]

        if self.task in ['ramr', 'raml', 'ftnl', 'ftnr']:
            task_names = [''.join(x) for x in itertools.product([self.task], ['1', '2'])]
        elif self.task in ['ram', 'ftn']:
            task_names = [''.join(x) for x in itertools.product([self.task], ['l', 'r'], ['1', '2'])]
        else:
            task_names = [self.task]

        cdtask = ld.commondescr[(ld.commondescr["task"].isin(task_names))
                                    & ~(ld.commondescr[outcome_var].isnull())]

        annotation = cdtask.iloc[self.keepind]

        return annotation["patient"].values


    @property
    def labels(self):
        return self.lab.values

    @labels.setter
    def labels(self, lab):
        self.lab = lab

    @property
    def shape(self):
        return self.data.shape[1:]

    def __len__(self):
        return self.data.shape[0]
