import os
import joblib
import synapseclient
import pandas as pd
import numpy as np

datadir = os.getenv('PARKINSON_DREAM_LDOPA_DATA')

class LDopa(object):
    def __init__(self, download_files = True, reload_ = False):

        self.synapselocation_training = 'syn10495809'
        self.synapselocation_test = 'syn10701954'
        self.downloadpath = os.path.join(datadir,"download")

        self.cachepath = os.path.join(self.downloadpath,
                            "tsv_file_map.pkl")

        if not os.path.exists(self.downloadpath):
            os.mkdir(self.downloadpath)

        if  reload_ or not os.path.exists(self.cachepath):
            self.download(download_files)
            joblib.dump((self.commondescr, self.file_map), self.cachepath)

        else:
            self.load()

    def load(self):
        self.commondescr, self.file_map = joblib.load(self.cachepath)

    def download(self, download_files = True):

        syn = synapseclient.Synapse()

        syn.login()

        selectstr = 'select * from {}'.format(self.synapselocation_training)
        results = syn.tableQuery(selectstr)

        df = results.asDataFrame()

        filemap = {}

        if download_files:
            prev_cache_loc = syn.cache.cache_root_dir
            syn.cache.cache_root_dir = self.downloadpath

            tsv_files = syn.downloadTableColumns(results, "dataFileHandleId")
            filemap.update(tsv_files)

            syn.cache.cache_root_dir = prev_cache_loc

        self.commondescr = df
        self.file_map = filemap

        ts_data = self.commondescr['dataFileHandleId'].apply(self.getRecordingStats)

        #self.commondescr['timestamp'] = self.commondescr['dataFileHandleId'].apply(self.get_start_timestamp)

        self.commondescr = self.commondescr.join(ts_data)
        self.commondescr['timestamp'] =  self.commondescr['timestamp'].apply(pd.to_datetime, unit='s')

        syn.logout()


    def loadfile(self, fid, return_timestamp = False):
        fid = str(int(fid))
        fname = self.file_map[fid]

        data = pd.read_csv(fname, sep='\t', index_col=None, skipinitialspace=True)

        data["time_in_task"] = data['timestamp'] - data['timestamp'].iloc[0]

        data.rename(columns={
            'GENEActiv_X': 'x', 'GENEActiv_Y': 'y', 'GENEActiv_Z': 'z', 'GENEActiv_Magnitude': 'm',
            'Pebble_X': 'x', 'Pebble_Y': 'y', 'Pebble_Z': 'z', 'Pebble_Magnitude': 'm',
            }, inplace=True)
        col_order = ['time_in_task', 'x', 'y', 'z', 'm']

        if return_timestamp:
            col_order += ['timestamp']

        return data[col_order]

    def getEntry(self, device, patient, session, task, visit):
        cd = self.commondescr
        e = cd[(cd['device'] == device)
               & (cd['patient'] == patient)
               & (cd['session'] == session)
               & (cd['task'] == task)
               & (cd['visit'] == visit)]
        if e.shape[0] != 1:
            raise Exception('Invalid number of entries found ({:d})'.format(e.shape[0]))

        return self.loadfile(e["dataFileHandleId"])

    def getRecordingStats(self, fid):
        rec = self.loadfile(fid, return_timestamp=True)

        missing = rec[['x', 'y', 'z']].isnull().sum()
        missing.rename({'x': 'missing_x', 'y': 'missing_y', 'z' : 'missing_z'}, inplace=True)
        missing_unique = rec[['x', 'y', 'z']].isnull().any(axis=1).sum()


        return pd.concat((
            pd.Series({
            'timestamp' : rec["timestamp"].iloc[0],
            'mean_interval': rec["timestamp"].diff().mean(),
            'length': rec["timestamp"].count(),
            'missing_unique' : missing_unique
        }), missing))


if __name__ == '__main__':
    ld = LDopa(reload_=True)
    #patient = '17_BOS'  # most affected side = left
    #visit = 1
    #session = 1

    #e = ld.getEntry('Pebble', patient, visit, 'drnkg', session)
    #print e["x"].dtype


