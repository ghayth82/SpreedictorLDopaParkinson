from numpydataset import NumpyDataset


class RawData(NumpyDataset):

    def getValues(self, df):
        M = df[self.columns].values
        shift = M.mean(axis=0)
        M -= shift

        return M


if __name__ == "__main__":
    rd = RawData("tre", "all", reload_=True, mode="training")
    assert len(rd) == 3440, "raw-tre-all-training len wrong"

    rd = RawData("tre", "all", reload_=True, mode="test")
    assert len(rd) == 5167, "raw-tre-all-test len wrong"

    rd = RawData("tre", "meta", reload_=True, mode="training")
    assert len(rd) == 3440, "raw-tre-all-training len wrong"

    rd = RawData("tre", "meta", reload_=True, mode="test")
    assert len(rd) == 5167, "raw-tre-all-test len wrong"
