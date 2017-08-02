import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from numpydataset import NumpyDataset

datadir = os.getenv('PARKINSON_DREAM_DATA')

class NonYDeviceMotion(NumpyDataset):
    def __init__(self, variant, limit = None, reload_ = False):
        self.npcachefile = os.path.join(datadir, 
                "nonydevicemotion_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration",
            "gravity", "rotationRate"], ["x","y","z"]))
        NumpyDataset.__init__(self, variant, limit, reload_)

    def getValues(self, df):
        df = df[(df.gravity_y>0.8) | (df.gravity_y<-0.8)]
        return df[[ "_".join(el) for \
            el in self.columns ]].values

class NonYDeviceMotionOutbound(NonYDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for outbound walk
    '''
    def __init__(self, limit = None):
        NonYDeviceMotion.__init__(self, "outbound", limit)

class NonYDeviceMotionRest(NonYDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for rest phase
    '''
    def __init__(self, limit = None):
        NonYDeviceMotion.__init__(self, "rest", limit)

class NonYDeviceMotionReturn(NonYDeviceMotion):
    '''
    Filtered Non-Y up or down device motion data for return walk
    '''
    def __init__(self, limit = None):
        NonYDeviceMotion.__init__(self, "return", limit)
