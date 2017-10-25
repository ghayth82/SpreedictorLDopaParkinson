from keras.layers import Input, Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.pooling import AveragePooling1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.layers.core import Dense, Dropout, Flatten, Activation
from functools import wraps


def model_conv_glob(data, paramdims):
    '''
    Conv1D:
        {} x {}, relu
        GlobPool
    '''
    input = Input(shape=data["input_1"].shape, name='input_1')
    layer = Conv1D(paramdims[0], kernel_size=(paramdims[1]), activation = 'relu')(input)
    output = GlobalAveragePooling1D()(layer)
    return input, output


def model_pool_conv_glob(data, paramdims):
    '''
    Conv1D:
        {} pool
        {} x {}, relu
        GlobPool
    '''
    input = Input(shape=data["input_1"].shape, name='input_1')
    layer = AveragePooling1D(paramdims[0])(input)
    layer = Conv1D(paramdims[1], kernel_size=(paramdims[2]), activation = 'relu')(layer)
    output = GlobalAveragePooling1D()(layer)
    return input, output

def model_conv_2l_glob(data, paramdims):
    '''
    Conv1D:
        {} x {}, relu
        {} pooling
        {} x
        GlobPool
    '''
    input = Input(shape=data['input_1'].shape, name='input_1')
    layer = Conv1D(paramdims[0], kernel_size=(paramdims[1]),
            activation = 'relu')(input)

    layer = BatchNormalization()(layer)

    layer = MaxPooling1D(pool_size=paramdims[2])(layer)

    layer = Conv1D(paramdims[3], kernel_size=(paramdims[4]),
            activation = 'relu')(layer)
    layer = BatchNormalization()(layer)
    output = GlobalAveragePooling1D()(layer)
    return input, output


def model_conv_3l_glob(data, paramdims):
    '''
    Conv1D:
        {} x {}, relu
        {} pooling
        {} x {}
        {} pooling
        {} Dense
        GlobPool
    '''
    input = Input(shape=data['input_1'].shape, name='input_1')
    layer = Conv1D(paramdims[0], kernel_size=(paramdims[1]),
            activation = 'relu')(input)
    layer = BatchNormalization()(layer)

    layer = MaxPooling1D(pool_size=paramdims[2])(layer)

    layer = Conv1D(paramdims[3], kernel_size=(paramdims[4]),
            activation = 'relu')(layer)
    layer = BatchNormalization()(layer)

    layer = MaxPooling1D(pool_size=paramdims[5])(layer)
    layer = Dense(paramdims[6])(layer)
    layer = Dropout(0.5)(layer)

    output = GlobalAveragePooling1D()(layer)
    return input, output


def model_logreg(data, paramdims):
    '''
    logistig regresscion:
        {}
        Flatten
    '''
    input = Input(shape=data["input_1"].shape, name='input_1')
    output = Flatten()(input)

    return input, output

modeldefs = {
    'conv_30_100': (model_conv_glob, (30,100)),
    'conv_30_200': (model_conv_glob, (30,200)),
    'conv_30_300': (model_conv_glob, (30,300)),
    'conv_10_200': (model_conv_glob, (10,200)),
    'conv_50_200': (model_conv_glob, (50,200)),
    'conv2l_30_300_10_20_30': (model_conv_2l_glob, (30,200,10,3,30)),
    'conv2l_50_300_10_20_30': (model_conv_2l_glob, (50,200,10,3,30)),
    'conv2l_50_300_10_40_30': (model_conv_2l_glob, (50,200,10,3,30)),
    'conv2l_30_300_10_40_30': (model_conv_2l_glob, (30,200,10,3,30)),
    'conv3l_50_300_10_20_30_10_10': (model_conv_3l_glob, (50,200,10,20,30, 10,10)),
    'conv3l_30_300_10_20_30_10_10': (model_conv_3l_glob, (30,200,10,20,30, 10,10)),
    'conv3l_30_300_10_40_30_10_10': (model_conv_3l_glob, (30,200,10,40,30, 10,10)),
    'conv3l_50_300_10_40_30_10_10': (model_conv_3l_glob, (50,200,10,40,30, 10,10)),
    'conv3l_70_300_10_20_30_10_10': (model_conv_3l_glob, (70,200,10,20,30, 10,10)),
    'conv3l_50_300_10_20_30_10_10': (model_conv_3l_glob, (50,200,10,20,30, 10,10)),
#    'poolconv_10_50_20': (model_pool_conv_glob, (10,50,20)),
#    'poolconv_10_30_20': (model_pool_conv_glob, (10,30,20)),
#    'poolconv_10_30_30': (model_pool_conv_glob, (10,30,30)),
#    'logreg' : (model_logreg, (0,)),
}
