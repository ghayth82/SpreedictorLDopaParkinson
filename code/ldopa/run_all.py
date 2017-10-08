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


    da = {}
    for k in dataset[comb[0]].keys():
        da[k] = get_dataset(dataset[comb[0]][k])
        if args.noise:
            da[k].transformData = da[k].transformDataNoise
        if args.rotate:
            da[k].transformData = da[k].transformDataRotate
        if args.flip:
            da[k].transformData = da[k].transformDataFlipSign
        if args.rofl:
            da[k].transformData = da[k].transformDataFlipRotate


    model = Classifier(da, modeldefs[comb[1]], name=name, epochs = args.epochs)

    if model.summaryExists() and not args.overwrite:
        print "{} exists, skipping".format(os.path.basename(model.summary_file))
    else:
        model.fit(args.noise|args.rotate|args.flip|args.rofl)

        #TODO: not yet implemented (save all LOOC models? or which ones? train model with all subjects?)
        #model.saveModel()
