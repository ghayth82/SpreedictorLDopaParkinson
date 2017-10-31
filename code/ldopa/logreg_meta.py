from datamanagement.rawdata import RawData
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

sub ="bra"

rd=RawData(sub,"meta")

X = rd.getData()
y = rd.labels

cl = LogisticRegression()

cl.fit(X,y)
score = cl.predict(X)
print("LogReg -{}- AUC={}".format(sub, roc_auc_score(y, score)))
print("LogReg -{}- acc={}".format(sub, accuracy_score(y, score)))

print('#'*85)
sub ="dys"

rd=RawData(sub,"meta")

X = rd.getData()
y = rd.labels

cl = LogisticRegression()

cl.fit(X,y)
score = cl.predict(X)
print("LogReg -{}- AUC={}".format(sub, roc_auc_score(y, score)))
print("LogReg -{}- acc={}".format(sub, accuracy_score(y, score)))

####################################
print('#'*85)

sub ="tre"

rd=RawData(sub,"meta")

X = rd.getData()
y = rd.labels

cl = LogisticRegression()

cl.fit(X,y)
#score = cl.decision_function(X)
score=np.argmax(cl.predict_proba(X),axis=1)

print("LogReg -{}- AUC={}".format(sub, accuracy_score(y, score)))
