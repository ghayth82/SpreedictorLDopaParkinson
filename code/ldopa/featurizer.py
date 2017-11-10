import os
import synapseclient
from modeldefs import modeldefs
from datamanagement.datasets import dataset, get_dataset
import itertools
from classifier import Classifier
import numpy as np
import pandas as pd


outputdir = os.getenv('PARKINSON_DREAM_LDOPA_DATA')


class Featurizer(object):
    submission_comb = {
            "tre1":('fh_0.5-tre-all', 'metatime_deep_conv_v2', 'allaug'),
            "tre2":('raw-tre-all', 'metatime_deep_conv_v2', 'allaug'),
            "bra1": ('raw-bra-all', 'metatime_deep_conv_v2', 'allaug_v2'),
            "bra2": ('fh_0.5-bra-all', 'metatime_deep_conv_v2', 'allaug'),
            "dys1": ('raw-dys-all', 'metatime_conv2l_70_200_10_50_30_20_10', 'allaug'),
            "dys2": ('raw-dys-all', 'metatime_deep_conv_v2', 'allaug')}
    synapse_id = {
        "tre1":9606376, "bra1": 9606378, "dys1": 9606377,
        "tre2":9606376, "bra2": 9606378, "dys2": 9606377,
}


    '''
    This class reuses the pretrained Classifiers
    and generates feature predictions on the given
    dataset
    '''

    def __init__(self):
        '''
        Init Featurizer
        '''
        summary_path = os.path.join(outputdir, "submission")
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        self.summary_path = summary_path



    def generateSubmission(self, subch, version):
        # determine

        sub_comb = self.submission_comb[subch]
        name = '.'.join(sub_comb[:2])
        name = '_'.join([name, sub_comb[2]])

        da = {}
        X = {}
        for k in dataset[sub_comb[0]].keys():
            da[k] = get_dataset(dataset[sub_comb[0]][k], mode="test")
            X[k] = da[k].getData()




        model = Classifier(da, modeldefs[sub_comb[1]],
            comb=sub_comb, name=name, epochs = 1)

        model.loadModel()
        features = model.featurize(X)

        filehandles = da['input_1'].filehandles

        print("Feature Dimension: {}".format(features.shape))

        pdfeatures = pd.DataFrame(data = features,
            index = pd.Index(data=filehandles, name="dataFileHandleId"),
            columns= list(['{}{}'.format(x,y) for x,y in
                    itertools.product(['feat'], range(features.shape[1]))]))

        pdfeatures.to_csv(os.path.join(self.summary_path,
                "submission_{}_{}.csv".format(subch, name)),
                sep = ",")
        print("Submission file written to {}".format(
            "submission_{}_{}.csv".format(subch, name)))

    def submit(self, subch, version):
        folderid = 'syn11389770'

        import synapseclient
        from synapseclient import File, Evaluation
        syn = synapseclient.login()

        sub_comb = self.submission_comb[subch]
        version = '.'.join(sub_comb[:2])
        version = '_'.join([version, sub_comb[2]])

        name = "submission_{}_{}.csv".format(subch, version)

        # upload the file to the synapse project folder
        submissionfile = File(os.path.join(self.summary_path, name),
            parent = folderid)
        submissionfile = syn.store(submissionfile)

        team_entity = syn.getTeam("Spreedictors")
        submission = syn.submit(evaluation = self.synapse_id[subch],
            entity = submissionfile, name = "Spreedictor_{}".format(name),
            team = team_entity)

        syn.logout()

if __name__ == "__main__":
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(description = \
            'Featurize the dataset and submit to challenge board', formatter_class = \
            RawTextHelpFormatter)
    parser.add_argument('features', choices = [ k for k in Featurizer.submission_comb.keys()],
            help = "Selection of features")

    parser.add_argument('--gen', dest="gen",
            default=False, action='store_true', help = "Generate features")

    parser.add_argument('--submit', dest="submit",
            default=False, action='store_true', help = "Submit feature set.")

    parser.add_argument('-version', dest='version', default = 'v1',
        help = "Version tag for the submission")

    args = parser.parse_args()
    fe = Featurizer()
    if args.gen:
        fe.generateSubmission(args.features, args.version)

    if args.submit:
        fe.submit(args.features, args.version)
