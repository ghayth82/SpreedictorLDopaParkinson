from keras.layers import Input, Dense, Dropout, Concatenate
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

def meta_l1(data, paramdims):
    """
    Metadata model
    Dense({}, {})
    """
    # second input
    input = Input(shape=data['input_1'].shape, name='input_1')
    output = Dense(paramdims[0], activation = paramdims[1])(input)

    return input, output


def metatime_conv_2l_glob(data, paramdims):
    '''
    Conv1D:
        {} x {}, relu
        {} pooling
        {} x
        GlobPool
    '''
    # first input
    input1 = Input(shape=data['input_1'].shape, name='input_1')
    tlayer = Conv1D(paramdims[0], kernel_size=(paramdims[1]),
            activation = 'relu')(input1)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[2])(tlayer)

    tlayer = Conv1D(paramdims[3], kernel_size=(paramdims[4]),
            activation = 'relu')(tlayer)
    tlayer = BatchNormalization()(tlayer)
    tlayer = GlobalAveragePooling1D()(tlayer)
    #tlayer = Dropout(0.5)(tlayer)

    # second input
    input2 = Input(shape=data['input_2'].shape, name='input_2')
    mlayer = Dense(paramdims[5], activation = 'relu')(input2)
    #mlayer = Dropout(0.5)(mlayer)

    # merge networks
    layer = Concatenate()([tlayer, mlayer])
    output = Dense(paramdims[6], activation = 'relu')(layer)

    return [input1, input2], output

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


def deep_conv(data, paramdims):

    # first input
    input1 = Input(shape=data['input_1'].shape, name='input_1')
    #1
    tlayer = Conv1D(paramdims[0], kernel_size=(paramdims[1]),
            activation = 'relu')(input1)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[2])(tlayer)

    #l2
    tlayer = Conv1D(paramdims[3], kernel_size=(paramdims[4]),
            activation = 'relu')(tlayer)
    tlayer = BatchNormalization()(tlayer)
    tlayer = MaxPooling1D(pool_size=paramdims[5])(tlayer)

    #l3
    tlayer = Conv1D(paramdims[6], kernel_size=(paramdims[7]),
            activation = 'relu')(tlayer)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[8])(tlayer)

    #l4
    tlayer = Conv1D(paramdims[9], kernel_size=(paramdims[10]),
            activation = 'relu')(tlayer)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[11])(tlayer)

    #l5
    tlayer = Conv1D(paramdims[12], kernel_size=(paramdims[13]),
            activation = 'relu')(tlayer)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[14])(tlayer)

    #l6
    tlayer = Conv1D(paramdims[15], kernel_size=(paramdims[16]),
            activation = 'relu')(tlayer)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[17])(tlayer)

    #l7
    tlayer = Conv1D(paramdims[18], kernel_size=(paramdims[19]),
            activation = 'relu')(tlayer)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[20])(tlayer)

    #l7
    tlayer = Conv1D(paramdims[21], kernel_size=(paramdims[22]),
            activation = 'relu')(tlayer)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[23])(tlayer)


    tlayer = GlobalAveragePooling1D()(tlayer)
    #tlayer = Dropout(0.5)(tlayer)

    # second input
    input2 = Input(shape=data['input_2'].shape, name='input_2')
    mlayer = Dense(paramdims[24], activation = 'relu')(input2)
    #mlayer = Dropout(0.5)(mlayer)

    # merge networks
    layer = Concatenate()([tlayer, mlayer])
    output = Dense(paramdims[25], activation = 'relu')(layer)

    return [input1, input2], output



def deep_conv_nometa(data, paramdims):

    # first input
    input1 = Input(shape=data['input_1'].shape, name='input_1')
    #1
    tlayer = Conv1D(paramdims[0], kernel_size=(paramdims[1]),
            activation = 'relu')(input1)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[2])(tlayer)

    #l2
    tlayer = Conv1D(paramdims[3], kernel_size=(paramdims[4]),
            activation = 'relu')(tlayer)
    tlayer = BatchNormalization()(tlayer)
    tlayer = MaxPooling1D(pool_size=paramdims[5])(tlayer)

    #l3
    tlayer = Conv1D(paramdims[6], kernel_size=(paramdims[7]),
            activation = 'relu')(tlayer)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[8])(tlayer)

    #l4
    tlayer = Conv1D(paramdims[9], kernel_size=(paramdims[10]),
            activation = 'relu')(tlayer)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[11])(tlayer)

    #l5
    tlayer = Conv1D(paramdims[12], kernel_size=(paramdims[13]),
            activation = 'relu')(tlayer)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[14])(tlayer)

    #l6
    tlayer = Conv1D(paramdims[15], kernel_size=(paramdims[16]),
            activation = 'relu')(tlayer)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[17])(tlayer)

    #l7
    tlayer = Conv1D(paramdims[18], kernel_size=(paramdims[19]),
            activation = 'relu')(tlayer)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[20])(tlayer)

    #l7
    tlayer = Conv1D(paramdims[21], kernel_size=(paramdims[22]),
            activation = 'relu')(tlayer)

    tlayer = BatchNormalization()(tlayer)

    tlayer = MaxPooling1D(pool_size=paramdims[23])(tlayer)


    tlayer = GlobalAveragePooling1D()(tlayer)
    #tlayer = Dropout(0.5)(tlayer)


    output = Dense(paramdims[24], activation = 'relu')(tlayer)

    return input1, output

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


def model_logreg_meta(data, paramdims):
    '''
    logistig regresscion:
        {}
        Flatten
    '''
    input = Input(shape=data["input_1"].shape, name='input_1')
    output = input

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
#    'conv_30_100': (model_conv_glob, (30,100)),
    'conv_30_200': (model_conv_glob, (30,200)),
#    'conv_30_300': (model_conv_glob, (30,300)),
#    'conv_10_200': (model_conv_glob, (10,200)),
#    'conv_50_200': (model_conv_glob, (50,200)),
#    'conv2l_30_300_10_20_30': (model_conv_2l_glob, (30,200,10,3,30)),
#    'conv2l_50_300_10_20_30': (model_conv_2l_glob, (50,200,10,3,30)),
    'conv2l_50_200_10_3_30': (model_conv_2l_glob, (50,200,10,3,30)),
    'conv2l_30_200_10_3_30': (model_conv_2l_glob, (30,200,10,3,30)),
#    'conv3l_50_300_10_20_30_10_10': (model_conv_3l_glob, (50,200,10,20,30, 10,10)),
#    'conv3l_30_300_10_20_30_10_10': (model_conv_3l_glob, (30,200,10,20,30, 10,10)),
#    'conv3l_30_300_10_40_30_10_10': (model_conv_3l_glob, (30,200,10,40,30, 10,10)),
#    'conv3l_50_300_10_40_30_10_10': (model_conv_3l_glob, (50,200,10,40,30, 10,10)),
#    'conv3l_70_300_10_20_30_10_10': (model_conv_3l_glob, (70,200,10,20,30, 10,10)),
#    'conv3l_50_300_10_20_30_10_10': (model_conv_3l_glob, (50,200,10,20,30, 10,10)),
#    'metatime_conv_2l_glob': (metatime_conv_2l_glob, (30,200,10,50,30, 20, 10)),
    'metatime_conv2l_70_200_10_50_30_20_10': (metatime_conv_2l_glob, (70,200,10,50,30, 20, 10)),
    'meta_l1_relu': (meta_l1, (10, 'relu')),
    'meta_l1_tanh': (meta_l1, (10, 'tanh')),
    'metatime_deep_conv': (deep_conv, (8, 3, 2,
                                        16, 4, 2,
                                        32, 3, 2,
                                        32, 3, 2,
                                        64, 3, 2,
                                        64, 4, 2,
                                        128, 3, 2,
                                        128, 4, 2,
                                        10, 20)),
    #'metatime_deep_conv_v2': (deep_conv, (16, 6, 2,
    #                                    16, 4, 2,
    #                                    32, 3, 2,
    #                                    32, 3, 2,
    #                                    64, 3, 2,
    #                                    64, 4, 2,
    #                                    128, 3, 2,
    #                                    128, 4, 2,
    #                                    10, 20)),
    'time_deep_conv': (deep_conv_nometa, (8, 3, 2,
                                       16, 4, 2,
                                       32, 3, 2,
                                       32, 3, 2,
                                       64, 3, 2,
                                       64, 4, 2,
                                       128, 3, 2,
                                       128, 4, 2,
                                       20)),
    'time_deep_conv_v2': (deep_conv_nometa, (16, 6, 2,
                                          16, 4, 2,
                                          32, 3, 2,
                                          32, 3, 2,
                                          64, 3, 2,
                                          64, 4, 2,
                                          128, 3, 2,
                                          128, 4, 2,
                                          20)),
#    'poolconv_10_50_20': (model_pool_conv_glob, (10,50,20)),
#    'poolconv_10_30_20': (model_pool_conv_glob, (10,30,20)),
#    'poolconv_10_30_30': (model_pool_conv_glob, (10,30,30)),
    'logregmeta' : (model_logreg_meta, (0,)),
}
