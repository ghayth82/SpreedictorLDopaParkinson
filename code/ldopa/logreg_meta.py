from datamanagement.rawdata import RawData
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rd=RawData("bra","meta")

X = rd.getData()
y = rd.labels

cl = LogisticRegression()

cl.fit(X,y)
score = cl.decision_function(X)

print("LogReg - AUC={}".format(roc_auc_score(y, score))
