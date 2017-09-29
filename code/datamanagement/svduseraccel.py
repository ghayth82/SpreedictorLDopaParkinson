import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from .numpydataset import NumpyDataset

datadir = os.getenv('PARKINSON_DREAM_DATA')

class SvdUserAccel(NumpyDataset):
    def __init__(self, variant, reload_ = False, training = True):
        self.npcachefile = os.path.join(datadir,
                "svduseraccel_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration"], \
                    ["x","y","z"]))
        NumpyDataset.__init__(self, "deviceMotion", variant, reload_, training)

    def getValues(self, df):
        M = df[[ "_".join(el) for \
            el in self.columns]].values.astype("float32")
        shift = M.mean(axis = 0)
        M -= shift
        U, s, V = np.linalg.svd(M, full_matrices = 0)
        return np.dot(U,np.diag(s))

#    def transformData(self, data):
#        return batchRandomRotation(data)

class SvdUserAccelOutbound(SvdUserAccel):
    '''
    SVD userAcceleration data for outbound walk
    '''
    def __init__(self, reload_ = False, training = True):
        SvdUserAccel.__init__(self, "outbound", reload_, training)

class SvdUserAccelRest(SvdUserAccel):
    '''
    SVD userAcceleration data for rest phase
    '''
    def __init__(self, reload_ = False, training = True):
        SvdUserAccel.__init__(self, "rest", reload_, training)

class SvdUserAccelReturn(SvdUserAccel):
    '''
    SVD userAcceleration data for return walk
    '''
    def __init__(self, reload_ = False, training = True):
        SvdUserAccel.__init__(self, "return", reload_, training)
