import itertools
from numpydataset import *
from rawdata import RawData
from filtereddata import FilteredData

def get_dataset_names(data_transform):
    return ['-'.join(x) for x in itertools.product([data_transform],
        ['tre'], tremor_tasks)] \
        + ['-'.join(x) for x in itertools.product([data_transform],
        ['dys'], dyskin_tasks)] \
        + ['-'.join(x) for x in itertools.product([data_transform], ['bra'],
        brakin_tasks)]


# dataset prefixes:
# - r               raw
# - fh_[cutoff]     high-pass filter, [cutoff] = cutoff frequency
# - fl_[cutoff]     low-pass filter, [cutoff] = cutoff frequency
# - fb_[low]_[high] band-pass filter, [low], [high] = cutoff frequencies
dataset_names = get_dataset_names('raw') + \
                get_dataset_names('fh_0.5') + \
                get_dataset_names('fb_0.5_20') + \
                get_dataset_names('fl_20')

dataset = dict(zip(dataset_names, [{'input_1' : x} for x in dataset_names]))

dataset = {}

# some additional dataset
####
##
# All data is used for the models that use metadata + timeseries
# for all available tasks per subchallenge
sub = ["tre", "bra", "dys"]
prep = ["raw", "fh_0.5", "fb_0.5_20"]

for s in sub:
    for p in prep:
        x = "{}-{}-all".format(p, s)
        m = "{}-{}-meta".format(p, s)
        #dataset_names = ["raw-tre-all", "raw-bra-all", "raw-dys-all"]
        d1 = {x:{'input_1' : x, 'input_2' : m}}
        dataset.update(d1)


# raw-bra-meta
#
# This dataset is used to test how well bradykinesia
# can be predicted only from meta data ( site, device, deviceside, ...)
dataset_names = ["raw-bra-meta", "raw-dys-meta", "raw-tre-meta"]

d1 = dict(zip(dataset_names, [{'input_1' : x} for x in dataset_names]))

dataset.update(d1)

# generate "mPower-like" dict from dataset names


def get_dataset(name, reload_ = False, mode = "training"):
    (data_transform, outcome, task) = name.split('-')

    if data_transform == 'raw':
        return RawData(outcome, task, reload_=reload_, mode=mode)
    elif data_transform[0] == 'f':
        filter_type = {'b' : 'band', 'h' : 'high', 'l' : 'low'}[data_transform[1]]
        freqs = [float(x) for x in data_transform.split('_')[1:]]

        return FilteredData(outcome, task, filter_type, freqs, reload_=reload_, mode=mode)
    else:
        raise Exception('Unknown data type:', data_transform)
