from .numpydataset import NumpyDataset
from scipy import signal
import numpy as np
import os

datadir = os.getenv('PARKINSON_DREAM_LDOPA_DATA')

class FilteredData(NumpyDataset):

    def __init__(self, outcome, task, filter_type, freqs, reload_=False):
        self.filter_type = filter_type
        self.freqs = freqs
        self.filter = self.createFilter(filter_type, freqs)

        self.npcachefile = os.path.join(datadir,
                                        "{}_{}_{}_{}_{}.pkl".format(
                                            self.__class__.__name__.lower(),
                                            filter_type,
                                            '-'.join(["{}".format(x) for x in freqs]),
                                            outcome,
                                            task))

        NumpyDataset.__init__(self, outcome, task, reload_)

    def getValues(self, df):
        M = df[self.columns].values

        return self.filterData(M)

    def filterData(self, M):

        filtered_x = signal.lfilter(self.filter[0], 1.0, M, axis=0)

        return filtered_x[(self.filter[1] - 1):,:]

    def createFilter(self, type, freqs, Fs=50.0):
        nyq_rate = Fs / 2.0
        width = 2.0 / nyq_rate
        ripple_db = 60.0

        N, beta = signal.kaiserord(ripple_db, width)

        pass_zero = (type=='low')
        if not pass_zero and (N % 2 == 0):
            N += 1

        taps = signal.firwin(N, np.array(freqs) / float(nyq_rate), window=('kaiser', beta), pass_zero=pass_zero)

        return (taps, N)
