
import os
import joblib
import itertools
import synapseclient
import pandas as pd
import numpy as np
from numpydataset import NumpyDataset

datadir = os.getenv('PARKINSON_DREAM_DATA')

class RawDeviceMotion(NumpyDataset):
    def __init__(self, variant, limit = None, reload_ = False):
        self.npcachefile = os.path.join(datadir, 
                "rawdevicemotion_{}.pkl".format(variant))

        self.columns = list(itertools.product(["userAcceleration",
            "gravity", "rotationRate"], ["x","y","z"]))

        NumpyDataset.__init__(self, variant, limit, reload_)

    def getValues(self, df):
        return df[[ "_".join(el) for \
            el in  self.columns ]].values

class RawDeviceMotionOutbound(RawDeviceMotion):
    '''
    Raw device motion data for outbound walk
    '''
    def __init__(self, limit = None):
        RawDeviceMotion.__init__(self, "outbound", limit)

class RawDeviceMotionRest(RawDeviceMotion):
    '''
    Raw device motion data for rest phase
    '''
    def __init__(self, limit = None):
        RawDeviceMotion.__init__(self, "rest", limit)

class RawDeviceMotionReturn(RawDeviceMotion):
    '''
    Raw device motion data for return walk
    '''
    def __init__(self, limit = None):
        RawDeviceMotion.__init__(self, "return", limit)
