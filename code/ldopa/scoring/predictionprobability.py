from __future__ import division
import numpy as np
import itertools


def pk(y_true, y_score):
    idx = np.argsort(y_score)

    y_score = y_score[idx]
    y_true = y_true[idx]

    pc = 0
    pd = 0
    pt = 0

    labels = np.sort(np.unique(y_true))
    for label_pair in list(itertools.combinations(labels, 2)):

        x = y_score[np.in1d(y_true, label_pair)]
        y = y_true[np.in1d(y_true, label_pair)]


        labels_n = np.where(y==label_pair[0])[0]
        labels_p = np.where(y==label_pair[1])[0]

        for i in labels_n:
            pc += np.sum(x[labels_p]  > x[i])
            pt += np.sum(x[labels_p] == x[i])
            pd += np.sum(x[labels_p]  < x[i])

    dyx = (pc-pd) / (pc+pd+pt)

    return (dyx+1)/2.0


if __name__ == "__main__":
    from sklearn import metrics

    y_true =  np.random.randint(0,5,200)
    #y_score = ( np.random.rand(200)>0.5)*1
    y_score = np.random.rand(200)*30

    y_true = np.array([0,1,2,3,4])
    y_score = np.array([4,3,2,1,0])
    print pk(y_true, y_score)
    #print metrics.roc_auc_score(y_true, y_score)


