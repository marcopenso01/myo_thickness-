import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
import configuration as config

from tfwrapper import utils



def max_pool_layer2d(x, kernel_size=(2, 2), strides=(2, 2), padding="SAME"):
    '''
    2D max pooling layer with standard 2x2 pooling as default
    '''
    pool_size = (kernel_size[0], kernel_size[1])
    strides = (strides[0], strides[1])
    return layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(x)


def avg_pool_layer2d(x, kernel_size=(2, 2), strides=(2, 2), padding="SAME"):
    '''
    2D average pooling layer with standard 2x2 pooling as default
    '''

    pool_size = (kernel_size[0], kernel_size[1])
    strides = (strides[0], strides[1])
    return layers.AveragePooling2D(pool_size=pool_size, strides=strides, padding=padding)(x)


def max_pool_layer3d(x, kernel_size=(2, 2, 2), strides=(2, 2, 2), padding="SAME"):
    '''
    3D max pooling layer with 2x2x2 pooling as default
    '''

    pool_size = (kernel_size[0], kernel_size[1], kernel_size[2])
    strides = (strides[0], strides[1], strides[2])
    return layers.MaxPool3D(pool_size=pool_size, strides=strides, padding=padding)(x)


def dropout_layer(bottom, name, training, keep_prob=0.5):
    '''
    Performs dropout on the activations of an input
    '''
    return layers.Dropout(rate=keep_prob)(bottom)


def batch_normalisation_layer(bottom):
    '''
    :return: Batch normalised activation
    '''
    return layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(bottom)


def upsample(tensor, rate=2):
    return layers.UpSampling2D(size=(rate,rate), interpolation='nearest')(tensor)


### FEED_FORWARD LAYERS ##############################################################################

def conv2D_layer(bottom,
                 kernel_size=(3, 3),
                 num_filters=32,
                 strides=(1, 1),
                 activation='linear',
                 padding="SAME",
                 weight_init='he_normal',
                 use_bias=True):
    '''
    Standard 2D convolutional layer
    bottom = input data
    '''
    return layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                         activation=activation, use_bias=use_bias, kernel_initializer=weight_init)(bottom)


# dense block
def dense_block(bottom,
                training,
                kernel_size=(3, 3),
                growth_rate=32,
                strides=(1, 1),
                activation=tf.keras.activations.relu,
                padding="SAME",
                weight_init='he_normal',
                n_layers=4):
    '''
    num_filters:= growth_rate
    MOre details here: https://towardsdatascience.com/paper-review-densenet-densely-connected-convolutional-networks-acf9065dfefb
    '''
    # Bottleneck Layer
    x = batch_normalisation_layer(bottom)
    x = activation(x)
    x = conv2D_layer(bottom=x,
                     kernel_size=(1, 1),
                     num_filters=128,
                     strides=strides,
                     activation='linear',
                     padding=padding,
                     weight_init=weight_init,
                     use_bias=False)
    x = dropout_layer(x)

    concat_feat = layers.concatenate([bottom, x], axis=-1)

    for i in range(1, n_layers):
        x = batch_normalisation_layer(concat_feat)
        x = activation(x)
        x = conv2D_layer(bottom=x,
                         kernel_size=(1, 1),
                         num_filters=128,
                         strides=strides,
                         activation='linear',
                         padding=padding,
                         weight_init=weight_init,
                         use_bias=False)
        x = dropout_layer(x)

        x = batch_normalisation_layer(x)
        x = activation(x)
        x = conv2D_layer(bottom=x,
                         kernel_size=kernel_size,
                         num_filters=growth_rate,
                         strides=strides,
                         activation='linear',
                         padding=padding,
                         weight_init=weight_init,
                         use_bias=False)
        x = dropout_layer(x)

        concat_feat = layers.concatenate([concat_feat, x], axis=-1)

    return concat_feat


def transition_layer(bottom,
                     training,
                     num_filters=32,
                     activation=tf.keras.activations.relu,
                     weight_init='he_normal',
                     pool=0):
    '''
    0: avg pool
    1: max pool
    '''
    x = batch_normalisation_layer(bottom)
    x = activation(x)
    x = conv2D_layer(bottom=x,
                     kernel_size=(1, 1),
                     num_filters=num_filters,
                     strides=(1, 1),
                     activation='linear',
                     padding="SAME",
                     weight_init=weight_init,
                     use_bias=False)
    x = dropout_layer(x)

    if pool == 0:
        x = avg_pool_layer2d(x)
    elif pool == 1:
        x = max_pool_layer2d(x)

    return x


def conv3D_layer(bottom,
                 kernel_size=(3, 3, 3),
                 num_filters=32,
                 strides=(1, 1, 1),
                 activation=tf.keras.activations.relu,
                 padding="SAME",
                 weight_init='he_normal',
                 use_bias=True):
    '''
    Standard 3D convolutional layer
    '''

    return layers.Conv3D(filters=num_filters, kernel_size=kernel_size, strides=strides,
                         padding=padding, activation=activation, use_bias=use_bias,
                         kernel_initializer=weight_init)(bottom)


def deconv2D_layer(bottom,
                   kernel_size=(4, 4),
                   num_filters=32,
                   strides=(2, 2),
                   activation='relu',
                   padding="SAME",
                   weight_init='he_normal',
                   use_bias=True):
    '''
    Standard 2D transpose (also known as deconvolution) layer.
    '''

    return layers.Conv2DTranspose(filters=num_filters, kernel_size=kernel_size, strides=strides, 
                                  padding=padding, use_bias=use_bias, activation=activation,
                                  kernel_initializer=weight_init)(bottom)


def deconv3D_layer(bottom,
                   kernel_size=(4, 4, 4),
                   num_filters=32,
                   strides=(2, 2, 2),
                   activation='relu',
                   padding="SAME",
                   weight_init='he_normal',
                   use_bias=True):
    '''
    Standard 3D transpose (also known as deconvolution) layer. 
    '''

    return layers.Conv3DTranspose(filters=num_filters, kernel_size=kernel_size, strides=strides, 
                                  padding=padding, use_bias=use_bias, activation=activation,
                                  kernel_initializer=weight_init)(bottom)


def conv2D_dilated_layer(bottom,
                         kernel_size=(3, 3),
                         num_filters=32,
                         strides=(1, 1),
                         rate=1,
                         activation='relu',
                         padding="SAME",
                         weight_init='he_normal',
                         use_bias=True):
    '''
    2D dilated convolution layer. This layer can be used to increase the receptive field of a network. 
    It is described in detail in this paper: Yu et al, Multi-Scale Context Aggregation by Dilated Convolutions, 
    2015 (https://arxiv.org/pdf/1511.07122.pdf) 
    '''

    return layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                         activation=activation, use_bias=use_bias, kernel_initializer=weight_init,
                         dilation_rate=(rate,rate))(bottom)


def dense_layer(bottom,
                hidden_units=512,
                activation='linear',
                weight_init='he_normal',
                use_bias=True):
    '''
    Dense fully connected layer
    '''

    return layers.Dense(units=hidden_units,
                        activation=activation,
                        kernel_initializer=weight_init,
                        use_bias=use_bias)(bottom)


# Squeeze and Excitation
'''
Let’s add parameters to each channel of a convolutional block so that the 
network can adaptively adjust the weighting of each feature map. 
As described in https://arxiv.org/pdf/1709.01507.pdf
'''

def se_block(tensor,
             ratio=16):
    init = tensor
    # number of input channels
    num_filters = tensor.get_shape().as_list()[-1]
    se_shape = (1, 1, num_filters)
    
    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(num_filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(num_filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    
    return layers.multiply([init, se])



# Selective Kernel 
'''
It is a process in which convolutional layers can adaptively adjust their 
receptive field (RF) sizes.
Advantage: Feature map with different receptive field (RF) in order to collect 
multi-scale spatial information.
As described in https://arxiv.org/pdf/1903.06586.pdf
'''

def selective_kernel_block(bottom,
                           num_filters=32,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           activation='relu',
                           padding="SAME",
                           weight_init='he_normal',
                           use_bias=False,
                           M=2,
                           r=16,
                           L=32):
    '''
    M: number of path
    r: number of parameters in the fuse operator
    G: controls the cardinality of each path
    '''
    filters = num_filters
    d = max(int(filters / r), L)
    #d = int(input_feature / 2)
    x = bottom
    xs = []
    for i in range(M):
        net = layers.conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                                use_bias=use_bias, kernel_initializer=weight_init, dilation_rate=1+i)(bottom)
        net = batch_normalisation_layer(net)
        net = activation(net)
        xs.append(net)
        
    U = layers.Add()(xs)
    
    gap = layers.GlobalAveragePooling2D()(U)
    fc = layers.Dense(d, kernel_initializer='he_normal', use_bias=False)(gap)

    att_vec = []
    for i in range(M):
        fcs = layers.Dense(filters, kernel_initializer='he_normal', use_bias=False)(fc)
        fcs_soft = layers.Softmax()(fcs)
        fea_V = layers.multiply([fcs_soft, x[i]])
        att_vec = att_vec.append(fea_V)
                                 
    y = layers.Add()(att_vec)

    return y


# Convolutional block attention module (CBAM)
'''
Contains the implementation of Convolutional Block Attention Module(CBAM) block.
As described in https://arxiv.org/abs/1807.06521.
'''

def conv_block_att_module(bottom,
                          kernel_size=(7, 7),
                          ratio=8,
                          activation=tf.nn.relu,
                          weight_init='he_normal'):
                                 
    attention_feature = channel_attention(bottom=bottom,
                                          ratio=ratio,
                                          activation=activation,
                                          weight_init=weight_init)
    attention_feature = spatial_attention(bottom=attention_feature,
                                          kernel_size=kernel_size,
                                          weight_init=weight_init)

    return attention_feature

                                 
def channel_attention(bottom,
                      ratio=8,
                      activation=tf.nn.relu,
                      weight_init='he_normal'):
                                 
    channel = bottom.get_shape().as_list()[-1]
    avg_pool = layers.GlobalAveragePooling2D()(bottom)
    avg_pool = layers.Dense(channel // ratio,
                            activation=activation,
                            weight_init=weight_init,
                            use_bias=True)(avg_pool)
    avg_pool = layers.Dense(channel,
                            activation=activation,
                            weight_init=weight_init,
                            use_bias=True)(avg_pool)
                                 
    max_pool = layers.GlobalMaxPooling2D()(bottom)
    max_pool = layers.Dense(channel // ratio,
                            activation=activation,
                            weight_init=weight_init,
                            use_bias=True)(max_pool)
    max_pool = layers.Dense(channel,
                            activation=activation,
                            weight_init=weight_init,
                            use_bias=True)(max_pool)
                                 
    cbam_feature = layers.Add()([avg_pool,max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    return layers.multiply([bottom, cbam_feature])


def spatial_attention(bottom,
                      kernel_size=(7, 7),
                      weight_init='he_normal'):
                                 
    avg_pool = tf.reduce_mean(bottom, axis=[3], keepdims=True)
    max_pool = tf.reduce_max(bottom, axis=[3], keepdims=True)

    concat = layers.Concatenate(axis=3)([avg_pool, max_pool])

    cbam_feature  = layers.Conv2D(filters = 1,
                                  kernel_size=kernel_size,
                                  strides=1,
                                  padding='same',
                                  activation='sigmoid',
                                  kernel_initializer=weight_init,
                                  use_bias=False)(concat)	

    return layers.multiply([bottom, cbam_feature])


# Attention gate
'''Attention, in the context of image segmentation, is a way to highlight 
only the relevant activations during training. This reduces the computational 
resources wasted on irrelevant activations, providing the network with better 
generalization power. Essentially, the network can pay “attention” to certain 
parts of the image.
As described in https://arxiv.org/pdf/1804.03999.pdf
'''
                                 
def attention_gate(down_tensor,
                   up_tensor,
                   channel,
                   activation='ReLU',
                   weight_init='he_normal'):
                                 
    '''
    Input
    ----------
        X: input tensor, i.e., key and value.
        g: gated tensor, i.e., query.
        channel: number of intermediate channel.
                 Oktay et al. (2018) did not specify (denoted as F_int).
                 intermediate channel is expected to be smaller than the input channel.
        activation: a nonlinear attnetion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is 'ReLU'.
        attention: 'add' for additive attention; 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
    '''
    x = layers.conv2D(channel, 1, use_bias=False, strides=(2, 2),
                      padding='valid', kernel_initializer=weight_init)(down_tensor)

    g = layers.conv2D(channel, 1, use_bias=False, strides=(1, 1),
                      padding='same', kernel_initializer=weight_init)(up_tensor)

    net = layers.Add()([g, x])
    net = layers.Activation('relu')(net)
    net = layers.conv2D(1, 1, strides=(1, 1), padding='same', kernel_initializer=weight_init)(net)
    net = layers.Activation('sigmoid')(net)
                                 
    net = layers.UpSampling2D(size=(2,2))(net)

    return layers.multiply([net, down_tensor])


def projection(bottom,
               kernel_size=(1, 1),
               num_filters=32,
               strides=(1, 1),
               activation='relu',
               padding="SAME",
               weight_init='he_normal',
               use_bias=False):
                                 
    x = layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(bottom)
    x = layers.Activation(activation)(x)
    x = layers.conv2D(num_filters, kernel_size, strides=strides, padding=padding,
                      kernel_initializer=weight_init, use_bias=use_bias)(x)

    return x


### BATCH_NORM SHORTCUTS #####################################################################################

def conv2D_layer_bn(bottom,
                    kernel_size=(3, 3),
                    num_filters=32,
                    strides=(1, 1),
                    activation='relu',
                    padding="SAME",
                    weight_init='he_normal',
                    use_bias=False):
    '''
    Shortcut for batch normalised 2D convolutional layer
    '''
    x = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, 
                      padding=padding, kernel_initializer=weight_init, use_bias=use_bias)(bottom)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    return x


def conv3D_layer_bn(bottom,
                    kernel_size=(3, 3, 3),
                    num_filters=32,
                    strides=(1, 1, 1),
                    activation='relu',
                    padding="SAME",
                    weight_init='he_normal',
                    use_bias=False):
    '''
    Shortcut for batch normalised 3D convolutional layer
    '''

    x = layers.Conv3D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      kernel_initializer=weight_init, use_bias=use_bias)(bottom)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    return x


def deconv2D_layer_bn(bottom,
                      kernel_size=(4, 4),
                      num_filters=32,
                      strides=(2, 2),
                      activation='relu',
                      padding="SAME",
                      weight_init='he_normal',
                      use_bias=False):
    '''
    Shortcut for batch normalised 2D transposed convolutional layer
    '''
    x = layers.Conv2DTranspose(filters=num_filters, kernel_size=kernel_size, strides=strides, 
                               padding=padding, use_bias=use_bias, kernel_initializer=weight_init)(bottom)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    return x


def deconv3D_layer_bn(bottom,
                      kernel_size=(4, 4, 4),
                      num_filters=32,
                      strides=(2, 2, 2),
                      activation='relu',
                      padding="SAME",
                      weight_init='he_normal',
                      use_bias=False):
    '''
    Shortcut for batch normalised 3D transposed convolutional layer
    '''
    x = layers.Conv3DTranspose(filters=num_filters, kernel_size=kernel_size, strides=strides, 
                               padding=padding, use_bias=use_bias, kernel_initializer=weight_init)(bottom)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    return x


def conv2D_dilated_layer_bn(bottom,
                            kernel_size=(3, 3),
                            num_filters=32,
                            strides=(1, 1),
                            rate=1,
                            activation='relu',
                            padding="SAME",
                            weight_init='he_normal',
                            use_bias=False):
    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''
    x = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, 
                      padding=padding, kernel_initializer=weight_init, use_bias=use_bias,
                      dilation_rate=rate)(bottom)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    return x


def residual_block(bottom,
                   kernel_size=(3, 3),
                   num_filters=32,
                   strides=(1, 1),
                   activation='relu',
                   padding='same',
                   weight_init='he_normal',
                   use_bias=True):
    '''
    As described in https://arxiv.org/pdf/1711.10684.pdf
    '''
    x = layers.BatchNormalization()(bottom)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=(strides[0], strides[0]), 
                      padding=padding, kernel_initializer=weight_init, use_bias=use_bias)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=(strides[1], strides[1]), 
                      padding=padding, kernel_initializer=weight_init, use_bias=use_bias)(x)
    
    shortcut = layers.Conv2D(filters=num_filters, kernel_size=1, strides=(strides[0], strides[0]), 
                             padding=padding, kernel_initializer=weight_init, 
                             use_bias=use_bias)(bottom)
    shortcut = layers.BatchNormalization()(shortcut)

    output = layers.Add()([shortcut, x])

    return output


def res_block_initial(bottom,
                      kernel_size=(3, 3),
                      num_filters=32,
                      strides=(1, 1),
                      activation='relu',
                      padding='same',
                      weight_init='he_normal',
                      use_bias=True):
    '''
    As described in https://arxiv.org/pdf/1711.10684.pdf
    '''
    x = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=(strides[0], strides[0]), 
                      padding=padding, kernel_initializer=weight_init, use_bias=use_bias)(bottom)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=(strides[1], strides[1]), 
                      padding=padding, kernel_initializer=weight_init, use_bias=use_bias)(x)
    shortcut = layers.Conv2D(filters=num_filters, kernel_size=1, strides=(1, 1), padding=padding, 
                             kernel_initializer=weight_init, use_bias=use_bias)(bottom)
    shortcut = layers.BatchNormalization()(shortcut)

    output = layers.Add()([shortcut, x])

    return output


def dense_layer_bn(bottom,
                   hidden_units=512,
                   activation='relu',
                   weight_init='he_normal'):

    x = layers.Dense(units=hidden_units, kernel_initializer=weight_init)(bottom)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    return x
