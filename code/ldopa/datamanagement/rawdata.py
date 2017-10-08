from .numpydataset import NumpyDataset


class RawData(NumpyDataset):

    def getValues(self, df):
        M = df[self.columns].values
        shift = M.mean(axis=0)
        M -= shift

        return M
