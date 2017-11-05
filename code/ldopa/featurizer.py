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
    submission_comb = {"tre":('raw-tre-all', 'metatime_deep_conv', 'allaug'),
            "bra": ('raw-bra-all', 'metatime_deep_conv', 'allaug'),
            "dys": ('raw-dys-all', 'metatime_deep_conv', 'allaug')}
    synapse_id = {"tre":9606376, "bra": 9606377, "dys": 9606378}


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

        da = {}
        X = {}
        for k in dataset[sub_comb[0]].keys():
            da[k] = get_dataset(dataset[sub_comb[0]][k], mode="test")
            X[k] = da[k].getData()



        name = '.'.join(sub_comb[:2])
        name = '_'.join([name, sub_comb[2]])

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
                "submission_{}_{}.csv".format(subch, version)),
                sep = ",")
        print("Submission file written to {}".format("submission_{}_{}.csv".format(subch, version)))

    def submit(self, subch, version):
        folderid = 'syn11389770'

        import synapseclient
        from synapseclient import File, Evaluation
        syn = synapseclient.login()

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
    parser.add_argument('--gen_tre_feat', dest="trefeat",
            default=False, action='store_true', help = "Generate features for tremor submission.")
    parser.add_argument('--gen_bra_feat', dest="brafeat",
            default=False, action='store_true', help = "Generate features for bradykinesia submission.")
    parser.add_argument('--gen_dys_feat', dest="dysfeat",
            default=False, action='store_true', help = "Generate features for dyskinesia submission.")


    parser.add_argument('--submit_tre', dest="submit_tre",
            default=False, action='store_true', help = "Submit tremor feature set.")
    parser.add_argument('--submit_bra', dest="submit_bra",
            default=False, action='store_true', help = "Submit bradykinesia feature set.")
    parser.add_argument('--submit_dys', dest="submit_dys",
            default=False, action='store_true', help = "Submit dyskinesia feature set.")

    parser.add_argument('-version', dest='version', default = 'v1',
        help = "Version tag for the submission")

    args = parser.parse_args()
    fe = Featurizer()
    if args.trefeat:
        fe.generateSubmission("tre", args.version)
    if args.brafeat:
        fe.generateSubmission("bra", args.version)
    if args.dysfeat:
        fe.generateSubmission("dys", args.version)

    if args.submit_tre:
        fe.submit("tre", args.version)
    if args.submit_bra:
        fe.submit("bra", args.version)
    if args.submit_dys:
        fe.submit("dys", args.version)
