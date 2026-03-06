#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    04-Mar-2026 14:21:11

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from BDLP20.customLayers.reshapeLayer import reshapeLayer

def create_model():
    Input = keras.Input(shape=(18583,21,1))
    Pathways = layers.Conv2D(8599, (18583,1), name="Pathways_")(Input)
    formattingLayer_1 = layers.Permute((3,2,1))(Input)
    depthcat = layers.Concatenate(axis=-1)([Pathways, formattingLayer_1])
    leakyrelu_1 = layers.LeakyReLU(alpha=0.010000)(depthcat)
    batchnorm_1 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_1_")(leakyrelu_1)
    formattingLayer_all = layers.Permute((3,2,1))(batchnorm_1)
    FirstData = layers.Conv2D(1, (1,21), name="FirstData_")(formattingLayer_all)
    formattingLayer_first = layers.Permute((3,2,1))(FirstData)
    RefData = layers.Conv2D(20, (1,21), name="RefData_")(formattingLayer_all)
    formattingLayer_ref = layers.Permute((3,2,1))(RefData)
    gmpool = layers.GlobalMaxPool2D(keepdims=True)(formattingLayer_ref)
    gapool = layers.GlobalAveragePooling2D(keepdims=True)(formattingLayer_ref)
    concat = layers.Concatenate(axis=2)([gmpool, gapool, formattingLayer_first])
    formattingLayer_2 = layers.Permute((3,2,1))(concat)
    fc_1 = layers.Reshape((-1,), name="fc_1_preFlatten1")(formattingLayer_2)
    fc_1 = layers.Dense(256, name="fc_1_")(fc_1)
    leakyrelu_2 = layers.LeakyReLU(alpha=0.010000)(fc_1)
    batchnorm_2 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_2_")(leakyrelu_2)
    dropout = layers.Dropout(0.500000)(batchnorm_2)
    fc_2 = layers.Dense(64, name="fc_2_")(dropout)
    leakyrelu_3 = layers.LeakyReLU(alpha=0.010000)(fc_2)
    batchnorm_3 = layers.BatchNormalization(epsilon=0.000010, name="batchnorm_3_")(leakyrelu_3)
    fc_3 = layers.Dense(16, name="fc_3_")(batchnorm_3)
    leakyrelu_4 = layers.LeakyReLU(alpha=0.010000)(fc_3)
    letentSpace = layers.BatchNormalization(epsilon=0.000010, name="letentSpace_")(leakyrelu_4)
    fc_4 = layers.Dense(1, name="fc_4_")(letentSpace)
    sigmoid = layers.Activation('sigmoid')(fc_4)

    model = keras.Model(inputs=[Input], outputs=[sigmoid])
    return model
