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
    return = layers.MaxPool3D(pool_size=pool_size, strides=strides, padding=padding)(x)


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
    return layers.UpSampling2D(size(rate,rate), interpolation='nearest')(tensor)


### FEED_FORWARD LAYERS ##############################################################################

def conv2D_layer(bottom,
                 kernel_size=(3, 3),
                 num_filters=32,
                 strides=(1, 1),
                 activation='linear',
                 padding="SAME",
                 weight_init='he_normal',
                 add_bias=True):
    '''
    Standard 2D convolutional layer
    bottom = input data
    '''
    return layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                         activation=activation, use_bias=add_bias, kernel_initializer=weight_init)(bottom)


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
                     add_bias=False)
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
                         add_bias=False)
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
                         add_bias=False)
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
                     add_bias=False)
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
                 add_bias=True):
    '''
    Standard 3D convolutional layer
    '''

    return layers.Conv3D(filters=num_filters, kernel_size=kernel_size, strides=strides,
                         padding=padding, activation=activation, use_bias=add_bias,
                         kernel_initializer=weight_init)(bottom)


def deconv2D_layer(bottom,
                   kernel_size=(4, 4),
                   num_filters=32,
                   strides=(2, 2),
                   output_shape=None,
                   activation='relu',
                   padding="SAME",
                   weight_init='he_normal',
                   add_bias=True):
    '''
    Standard 2D transpose (also known as deconvolution) layer.
    '''

    return layers.Conv2DTranspose(filters=num_filters, kernel_size=kernel_size, strides=strides, 
                                  padding=padding, use_bias=add_bias, activation=activation,
                                  kernel_initializer=weight_init)(bottom)


def deconv3D_layer(bottom,
                   kernel_size=(4, 4, 4),
                   num_filters=32,
                   strides=(2, 2, 2),
                   output_shape=None,
                   activation='relu',
                   padding="SAME",
                   weight_init='he_normal',
                   add_bias=True):
    '''
    Standard 3D transpose (also known as deconvolution) layer. 
    '''

    return layers.Conv3DTranspose(filters=num_filters, kernel_size=kernel_size, strides=strides, 
                                  padding=padding, use_bias=add_bias, activation=activation,
                                  kernel_initializer=weight_init)(bottom)


def conv2D_dilated_layer(bottom,
                         kernel_size=(3, 3),
                         num_filters=32,
                         rate=1,
                         activation='relu',
                         padding="SAME",
                         weight_init='he_normal',
                         add_bias=True):
    '''
    2D dilated convolution layer. This layer can be used to increase the receptive field of a network. 
    It is described in detail in this paper: Yu et al, Multi-Scale Context Aggregation by Dilated Convolutions, 
    2015 (https://arxiv.org/pdf/1511.07122.pdf) 
    '''

    return layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                         activation=activation, use_bias=add_bias, kernel_initializer=weight_init,
                         dilation_rate=(rate,rate))(bottom)


def dense_layer(bottom,
                hidden_units=512,
                activation='linear',
                weight_init='he_normal',
                add_bias=True):
    '''
    Dense fully connected layer
    '''

    return layers.Dense(units=hidden_units,
                        activation=activation,
                        kernel_initializer=weight_init,
                        use_bias=add_bias)(bottom)


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
                           training,
                           num_filters=32,
                           kernel_size=(3, 3),
                           strides=(1, 1),
                           activation='relu',
                           padding="SAME",
                           weight_init='he_normal',
                           M=2,
                           r=16):
    '''
    M: number of path
    r: number of parameters in the fuse operator
    G: controls the cardinality of each path
    '''
    filters = num_filters
    d = max(int(filters / r), 32)
    #d = int(input_feature / 2)
    x = bottom

    for i in range(M):
        net = layers.conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                            use_bias=False, kernel_initializer=weight_init, dilation_rate=1+i)(bottom)
        net = batch_normalisation_layer(net)
        net = activation(net)

        if i == 0:
            U = net
        else:
            U = layers.add(U, net)
            
    gap = layers.GlobalAveragePooling2D()(U)
    fc = layers.Dense(d, kernel_initializer='he_normal', use_bias=False)(gap)
    fcs = fc

    for _ in range(M):
        fcs = layers.Dense(input_feature, kernel_initializer='he_normal', use_bias=False)(fcs)
        if _ == 0:
            att_vec = fcs
        else:
            att_vec = layers.add(att_vec, fcs)
    se_shape(1,1,input_feature)
    att_vec = layers.Reshape(se_shape)(se)
    att_vec_softmax = layers.Softmax()(att_vec)
    fea_v = layers.multiply([fea_U, att_vec_softmax])

    return fea_v


# Convolutional block attention module (CBAM)
'''
Contains the implementation of Convolutional Block Attention Module(CBAM) block.
As described in https://arxiv.org/abs/1807.06521.
'''


def conv_block_att_module(bottom,
                          name,
                          kernel_size=(7, 7),
                          ratio=8,
                          activation=tf.nn.relu,
                          weight_init='he_normal'):
    attention_feature = channel_attention(bottom=bottom,
                                          name=name,
                                          ratio=ratio,
                                          activation=activation,
                                          weight_init=weight_init)

    attention_feature = spatial_attention(bottom=attention_feature,
                                          name=name,
                                          kernel_size=kernel_size,
                                          activation=activation,
                                          weight_init=weight_init)

    return attention_feature


def channel_attention(bottom,
                      name,
                      ratio=8,
                      activation=tf.nn.relu,
                      weight_init='he_normal'):
    channel = bottom.get_shape().as_list()[-1]
    avg_pool = tf.reduce_mean(bottom, [1, 2], keepdims=True)
    avg_pool = dense_layer(bottom=avg_pool,
                           name=name + '_mpl0',
                           hidden_units=channel // ratio,
                           activation=activation,
                           weight_init=weight_init)
    avg_pool = dense_layer(bottom=avg_pool,
                           name=name + '_mpl1',
                           hidden_units=channel,
                           activation=activation,
                           weight_init=weight_init)
    max_pool = tf.reduce_max(bottom, axis=[1, 2], keepdims=True)
    max_pool = dense_layer(bottom=max_pool,
                           name=name + '_mpl2',
                           hidden_units=channel // ratio,
                           activation=activation,
                           weight_init=weight_init)
    max_pool = dense_layer(bottom=max_pool,
                           name=name + '_mpl3',
                           hidden_units=channel,
                           activation=activation,
                           weight_init=weight_init)
    scale = tf.add(avg_pool, max_pool)
    scale = tf.math.sigmoid(scale)

    return tf.multiply(bottom, scale)


def spatial_attention(bottom,
                      name,
                      kernel_size=(7, 7),
                      activation=tf.nn.relu,
                      weight_init='he_normal'):
    avg_pool = tf.reduce_mean(bottom, axis=[3], keepdims=True)
    max_pool = tf.reduce_max(bottom, axis=[3], keepdims=True)

    concat = tf.concat([avg_pool, max_pool], 3)

    concat = conv2D_layer(bottom=concat,
                          name=name + '_spatial',
                          kernel_size=kernel_size,
                          num_filters=1,
                          strides=(1, 1),
                          activation=tf.identity,
                          padding="SAME",
                          weight_init=weight_init,
                          add_bias=False)

    concat = tf.math.sigmoid(concat)

    return tf.multiply(bottom, concat)


# Attention gate
'''Attention, in the context of image segmentation, is a way to highlight 
only the relevant activations during training. This reduces the computational 
resources wasted on irrelevant activations, providing the network with better 
generalization power. Essentially, the network can pay “attention” to certain 
parts of the image.
As described in https://arxiv.org/pdf/1804.03999.pdf
'''


def attention(down_tensor,
              name,
              up_tensor,
              weight_init='he_normal'):
    channel = down_tensor.get_shape().as_list()[-1]

    x = conv2D_layer(bottom=down_tensor,
                     name=name + '_g',
                     kernel_size=(1, 1),
                     num_filters=channel,
                     strides=(2, 2),
                     activation=tf.identity,
                     padding="VALID",
                     weight_init=weight_init,
                     add_bias=False)

    g = conv2D_layer(bottom=up_tensor,
                     name=name + '_x',
                     kernel_size=(1, 1),
                     num_filters=channel,
                     strides=(1, 1),
                     activation=tf.identity,
                     padding="SAME",
                     weight_init=weight_init,
                     add_bias=False)

    net = tf.add(g, x)
    net = tf.nn.relu(net)
    net = conv2D_layer(bottom=net,
                       name=name,
                       kernel_size=(1, 1),
                       num_filters=1,
                       strides=(1, 1),
                       activation=tf.identity,
                       padding="SAME",
                       weight_init=weight_init,
                       add_bias=False)
    net = tf.math.sigmoid(net)
    net = tf.image.resize_bilinear(net, (tf.shape(net)[1] * 2, tf.shape(net)[2] * 2))

    return tf.multiply(net, down_tensor)


def projection(bottom,
               name,
               training,
               kernel_size=(1, 1),
               num_filters=32,
               strides=(1, 1),
               activation=tf.nn.relu,
               padding="SAME",
               weight_init='he_normal'):
    conv_bn = batch_normalisation_layer(bottom, name + '_bn', training)

    act = activation(conv_bn)

    conv = conv2D_layer(bottom=act,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False)

    return conv


### BATCH_NORM SHORTCUTS #####################################################################################

def conv2D_layer_bn(bottom,
                    name,
                    training,
                    kernel_size=(3, 3),
                    num_filters=32,
                    strides=(1, 1),
                    activation=tf.nn.relu,
                    padding="SAME",
                    weight_init='he_normal'):
    '''
    Shortcut for batch normalised 2D convolutional layer
    '''

    conv = conv2D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training)

    act = activation(conv_bn)

    return act


def conv3D_layer_bn(bottom,
                    name,
                    training,
                    kernel_size=(3, 3, 3),
                    num_filters=32,
                    strides=(1, 1, 1),
                    activation=tf.nn.relu,
                    padding="SAME",
                    weight_init='he_normal'):
    '''
    Shortcut for batch normalised 3D convolutional layer
    '''

    conv = conv3D_layer(bottom=bottom,
                        name=name,
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=strides,
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=False)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training)

    act = activation(conv_bn)

    return act


def deconv2D_layer_bn(bottom,
                      name,
                      training,
                      kernel_size=(4, 4),
                      num_filters=32,
                      strides=(2, 2),
                      output_shape=None,
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal'):
    '''
    Shortcut for batch normalised 2D transposed convolutional layer
    '''

    deco = deconv2D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=tf.identity,
                          padding=padding,
                          weight_init=weight_init,
                          add_bias=False)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training)

    act = activation(deco_bn)

    return act


def deconv3D_layer_bn(bottom,
                      name,
                      training,
                      kernel_size=(4, 4, 4),
                      num_filters=32,
                      strides=(2, 2, 2),
                      output_shape=None,
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal'):
    '''
    Shortcut for batch normalised 3D transposed convolutional layer
    '''

    deco = deconv3D_layer(bottom=bottom,
                          name=name,
                          kernel_size=kernel_size,
                          num_filters=num_filters,
                          strides=strides,
                          output_shape=output_shape,
                          activation=tf.identity,
                          padding=padding,
                          weight_init=weight_init,
                          add_bias=False)

    deco_bn = batch_normalisation_layer(deco, name + '_bn', training=training)

    act = activation(deco_bn)

    return act


def conv2D_dilated_layer_bn(bottom,
                            name,
                            training,
                            kernel_size=(3, 3),
                            num_filters=32,
                            rate=1,
                            activation=tf.nn.relu,
                            padding="SAME",
                            weight_init='he_normal'):
    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''

    conv = conv2D_dilated_layer(bottom=bottom,
                                name=name,
                                kernel_size=kernel_size,
                                num_filters=num_filters,
                                rate=rate,
                                activation=tf.identity,
                                padding=padding,
                                weight_init=weight_init,
                                add_bias=False)

    conv_bn = batch_normalisation_layer(conv, name + '_bn', training=training)

    act = activation(conv_bn)

    return act


def residual_block(bottom,
                   name,
                   training,
                   kernel_size=(3, 3),
                   num_filters=32,
                   strides=[1, 1],
                   activation=tf.nn.relu,
                   padding="SAME",
                   weight_init='he_normal'):
    '''
    As described in https://arxiv.org/pdf/1711.10684.pdf
    '''

    x = batch_normalisation_layer(bottom, name + '_bn1', training)

    x = activation(x)

    res1 = conv2D_layer(bottom=x,
                        name=name + '_1',
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=(strides[0], strides[0]),
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=True)

    res1 = batch_normalisation_layer(res1, name + '_bn2', training)

    res1 = activation(res1)

    res2 = conv2D_layer(bottom=res1,
                        name=name + '_2',
                        kernel_size=kernel_size,
                        num_filters=num_filters,
                        strides=(strides[1], strides[1]),
                        activation=tf.identity,
                        padding=padding,
                        weight_init=weight_init,
                        add_bias=True)

    shortcut = conv2D_layer(bottom=bottom,
                            name=name + '_shortcut',
                            kernel_size=(1, 1),
                            num_filters=num_filters,
                            strides=(strides[0], strides[0]),
                            activation=tf.identity,
                            padding=padding,
                            weight_init=weight_init,
                            add_bias=True)

    shortcut = batch_normalisation_layer(shortcut, name + '_bn3', training)

    output = tf.add(shortcut, res2)

    return output


def res_block_initial(bottom,
                      name,
                      training,
                      kernel_size=(3, 3),
                      num_filters=32,
                      strides=[1, 1],
                      activation=tf.nn.relu,
                      padding="SAME",
                      weight_init='he_normal'):
    '''
    As described in https://arxiv.org/pdf/1711.10684.pdf
    '''

    x = conv2D_layer(bottom=bottom,
                     name=name + '_1',
                     kernel_size=kernel_size,
                     num_filters=num_filters,
                     strides=(strides[0], strides[0]),
                     activation=tf.identity,
                     padding=padding,
                     weight_init=weight_init,
                     add_bias=True)

    x = batch_normalisation_layer(x, name + '_bn1', training)

    x = activation(x)

    x = conv2D_layer(bottom=x,
                     name=name + '_2',
                     kernel_size=kernel_size,
                     num_filters=num_filters,
                     strides=(strides[1], strides[1]),
                     activation=tf.identity,
                     padding=padding,
                     weight_init=weight_init,
                     add_bias=True)

    shortcut = conv2D_layer(bottom=bottom,
                            name=name + '_shortcut',
                            kernel_size=(1, 1),
                            num_filters=num_filters,
                            strides=(1, 1),
                            activation=tf.identity,
                            padding=padding,
                            weight_init=weight_init,
                            add_bias=True)

    shortcut = batch_normalisation_layer(shortcut, name + '_bn2', training)

    output = tf.add(shortcut, x)

    return output


def dense_layer_bn(bottom,
                   name,
                   training,
                   hidden_units=512,
                   activation=tf.nn.relu,
                   weight_init='he_normal'):
    '''
    Shortcut for batch normalised 2D dilated convolutional layer
    '''

    linact = dense_layer(bottom=bottom,
                         name=name,
                         hidden_units=hidden_units,
                         activation=tf.identity,
                         weight_init=weight_init)

    batchnorm = batch_normalisation_layer(linact, name + '_bn', training=training)
    act = activation(batchnorm)

    return act


### VARIABLE INITIALISERS ####################################################################################

def get_weight_variable(shape, name=None, regularize=True, **kwargs):
    type = config.weight_init
    initialise_from_constant = False
    if type == 'xavier_uniform':
        initial = glorot_uniform()
    elif type == 'xavier_normal':
        initial = glorot_normal()
    elif type == 'he_normal':
        initial = he_normal()
    elif type == 'he_uniform':
        initial = he_uniform()
    elif type == 'simple':
        stddev = kwargs.get('stddev', 0.02)
        initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
        initialise_from_constant = True
    elif type == 'bilinear':
        weights = _bilinear_upsample_weights(shape)
        initial = tf.constant(weights, shape=shape, dtype=tf.float32)
        initialise_from_constant = True
    else:
        raise ValueError('Unknown initialisation requested: %s' % type)
    if name is None:  # This keeps to option open to use unnamed Variables
        weight = tf.Variable(initial)
    else:
        if initialise_from_constant:
            weight = tf.compat.v1.get_variable(name, initializer=initial)
        else:
            weight = tf.compat.v1.get_variable(name, shape=shape, initializer=initial)

    if regularize:
        tf.compat.v1.add_to_collection('weight_variables', weight)

    return weight


def get_bias_variable(shape, name=None, init_value=0.0):
    initial = tf.constant(init_value, shape=shape, dtype=tf.float32)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.compat.v1.get_variable(name, initializer=initial)


def _upsample_filt(size):
    '''
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def _bilinear_upsample_weights(shape):
    '''
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    '''

    if not shape[0] == shape[1]: raise ValueError('kernel is not square')
    if not shape[2] == shape[3]: raise ValueError('input and output featuremaps must have the same size')

    kernel_size = shape[0]
    num_feature_maps = shape[2]

    weights = np.zeros(shape, dtype=np.float32)
    upsample_kernel = _upsample_filt(kernel_size)

    for i in range(num_feature_maps):
        weights[:, :, i, i] = upsample_kernel

    return weights


def _add_summaries(op, weights, biases):
    # Tensorboard variables
    tf.compat.v1.summary.histogram(weights.name, weights)
    if biases is not None:
        tf.compat.v1.summary.histogram(biases.name, biases)
    tf.compat.v1.summary.histogram(op.op.name + '/activations', op)
