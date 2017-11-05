#!/home/wkopp/anaconda2/bin/python
import itertools
import synapseclient
from modeldefs import modeldefs
from datamanagement.datasets import dataset, get_dataset
from classifier import Classifier
import numpy as np
import os

import argparse
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(description = \
        'Run all (or selection) of parkinson`s disease prediciton models.', formatter_class = \
        RawTextHelpFormatter)
parser.add_argument('-df', dest='datafilter', nargs = '*', default = [''],
        help = "Filter for datasets")
parser.add_argument('-mf', dest="modelfilter", nargs = '*',
        default = [''], help = "Filter for model definitions")

parser.add_argument('--overwrite', dest="overwrite", action='store_true',
        default=False, help = "Overwrite existing analyses")

parser.add_argument('--epochs', dest="epochs", type=int,
        default = 30, help = "Number of epochs")

parser.add_argument('--noise', dest="noise", action='store_true',
        default=False, help = "Augment with gaussian noise")

parser.add_argument('--rotate', dest="rotate", action='store_true',
        default=False, help = "Augment with random rotations")

parser.add_argument('--flip', dest="flip", action='store_true',
        default=False, help = "Augment by flipping the sign")
parser.add_argument('--rofl', dest="rofl", action='store_true',
        default=False, help = "Augment by flipping the sign and rotating")
parser.add_argument('--allaug', dest="allaug",
        default=False, action='store_true', help = "Data augmentation with all options")

args = parser.parse_args()
print(args.datafilter)
print(args.modelfilter)

import re

all_combinations = list(itertools.product(dataset, modeldefs))

for comb in all_combinations:

    x  = [ type(re.search(d, comb[0])) != type(None) for d in args.datafilter]
    if not np.any(x):
        continue
    x  = [ type(re.search(d, comb[1])) != type(None) for d in args.modelfilter]
    if not np.any(x):
        continue

    #name = '.'.join([args.data, args.model])

    print("Running {}-{}".format(comb[0],comb[1]))

    name = '.'.join(comb)
    #continue

    if args.noise:
        name = '_'.join([name, "aug"])
    print("--rotate {}".format(args.rotate))
    if args.rotate:
        name = '_'.join([name, "rot"])
    print("--flip {}".format(args.flip))
    if args.flip:
        name = '_'.join([name, "flip"])
    print("--rofl {}".format(args.rofl))
    if args.rofl:
        name = '_'.join([name, "rofl"])
    print("--rofl {}".format(args.rofl))
    if args.allaug:
        name = '_'.join([name, "allaug"])
    print(name)


    da = {}
    for k in dataset[comb[0]].keys():
        da[k] = get_dataset(dataset[comb[0]][k])
        if k != 'input_1':
            # quick-hack: for meta + timeseries
            # 'input_1' would be the timeseries
            # and 'input_2' the metadataset, which does not require transformData
            continue
        if args.noise:
            comb += ("noise",)
            da[k].transformData = da[k].transformDataNoise
        if args.rotate:
            comb += ("rotate",)
            da[k].transformData = da[k].transformDataRotate
        if args.flip:
            comb += ("flip",)
            da[k].transformData = da[k].transformDataFlipSign
        if args.rofl:
            comb += ("rofl",)
            da[k].transformData = da[k].transformDataFlipRotate
        if args.allaug:
            comb += ("allaug",)
            da[k].transformData = da[k].transformDataAll


    model = Classifier(da, modeldefs[comb[1]], comb=list(comb), name=name, epochs = args.epochs)

    if model.summaryExists() and not args.overwrite:
        print "{} exists, skipping".format(os.path.basename(model.summary_file))
    else:
        model.fit(args.noise|args.rotate|args.flip|args.rofl)

        model.saveModel()
