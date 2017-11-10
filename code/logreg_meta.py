from datamanagement.rawdata import RawData
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from scoring.predictionprobability import pk
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from classifier import generate_fit_data
import keras.backend as K
import tensorflow as tf

def logreg_sklearn(rd):
    X = rd.getData()
    y = rd.labels

    cl = LogisticRegression()

    cl.fit(X,y)
    score = cl.decision_function(X)
    print("#"*85)
    print("LogReg AUC={}".format(roc_auc_score(y, score)))
    score = cl.predict(X)
    print("LogReg acc={}".format(accuracy_score(y, score)))
    print("#"*85)

def logreg_tensorflow(rd):

    X = rd.getData()
    y = rd.labels

    #input
    x = tf.placeholder(tf.float32, [None, X.shape[1]], name="X")

    def sig_layer(input, kernel_shape, bias_shape):
        # Create variable named "weights".
        weights = tf.get_variable("weights", kernel_shape,
            initializer=tf.random_normal_initializer())
        # Create variable named "biases".
        biases = tf.get_variable("biases", bias_shape,
            initializer=tf.constant_initializer(0.0))
        return tf.nn.sigmoid(tf.matmul(input, weights) + biases)

    plabel = sig_layer(x, [X.shape[1],1], (1,))

    label = tf.placeholder(tf.float32, [None, 1], name="y")

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(label * tf.log(plabel) + \
            (1-label)*tf.log(1-plabel), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    for _ in range(1000):
        sess.run(train_step, feed_dict={x:X.astype("float32"),
                label:y[:,np.newaxis].astype("float32")})

    p = sess.run(plabel, {x:X.astype("float32")})

    print("#"*85)
    print("TF-LogReg AUC={}".format(roc_auc_score(y, p)))
    print("#"*85)

def logreg_keras_fit(rd):
    input = Input(shape=rd.shape, name="input_1")
    output = Dense(1, activation="sigmoid")(input)
    model = Model(input, output)
    model.compile(loss="binary_crossentropy", optimizer="adadelta")

    model.fit(X, y, epochs=1000, batch_size=100)
    #model.fit(x=rd.getData(), y=rd.labels, epochs=1000, batch_size=100)

    pkeras=model.predict(rd.getData())

    print("#"*85)
    print("keras - AUC={}".format(roc_auc_score(rd.labels, pkeras)))
    print("#"*85)

def logreg_keras_fitgen(rd):
    input = Input(shape=rd.shape, name="input_1")
    output = Dense(1, activation="sigmoid")(input)
    model = Model(input, output)
    model.compile(loss="binary_crossentropy", optimizer="rmsprop")

    bs=100
    model.fit_generator(generate_fit_data({'input_1':rd},
            range(len(rd)), np.ones((len(rd),)), bs, augment=False),
            steps_per_epoch = len(rd)//bs + \
                    (1 if len(rd)%bs > 0 else 0),
                epochs = 500, use_multiprocessing = False)

    pkeras=model.predict(rd.getData())

    print("#"*85)
    print("keras - AUC={}".format(roc_auc_score(rd.labels, pkeras)))
    print("#"*85)

if __name__ == "__main__":

    sub ="bra"

    rd=RawData(sub,"meta")
    logreg_keras_fitgen(rd)
    logreg_sklearn(rd)

    sub ="dys"
    rd=RawData(sub,"meta")
    logreg_keras_fitgen(rd)
    logreg_sklearn(rd)

    if False:
        #not needed for now
        sub ="tre"

        rd=RawData(sub,"meta")

        X = rd.getData().astype("float32")
        y = rd.labels.astype("float32")

        cl = LogisticRegression()

        cl.fit(X,y)
        #score = cl.decision_function(X)
        score=np.argmax(cl.predict_proba(X),axis=1)

        print("LogReg -{}- PK={}".format(sub, pk(y, score)))
        print("LogReg -{}- acc={}".format(sub, accuracy_score(y, score)))


        #######################################
        #
        print("#"*85)
        # for bra: AUC=0.79884798050607642
        # therefore, it agrees with sklearn



        K.clear_session()

