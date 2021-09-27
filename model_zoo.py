import tensorflow as tf
from tensorflow.keras import layers 
from tensorflow.keras import Model 
from tensorflow.keras.layers import *
from tfwrapper import layers as lr

#same
def unet2D_same(input_tensor, nlabels):
    
    inputs = layers.InputLayer(input_shape=(input_tensor.shape[1], input_tensor.shape[2], 1))
    conv1_1 = lr.conv2D_layer_bn(inputs, num_filters=64)
    conv1_2 = lr.conv2D_layer_bn(conv1_1, num_filters=64)
    pool1 = lr.max_pool_layer2d(conv1_2)

    conv2_1 = lr.conv2D_layer_bn(pool1, num_filters=128)
    conv2_2 = lr.conv2D_layer_bn(conv2_1, num_filters=128)
    pool2 = lr.max_pool_layer2d(conv2_2)

    conv3_1 = lr.conv2D_layer_bn(pool2, num_filters=256)
    conv3_2 = lr.conv2D_layer_bn(conv3_1, num_filters=256)
    pool3 = lr.max_pool_layer2d(conv3_2)

    conv4_1 = lr.conv2D_layer_bn(pool3, num_filters=512)
    conv4_2 = lr.conv2D_layer_bn(conv4_1, num_filters=512)
    pool4 = lr.max_pool_layer2d(conv4_2)

    conv5_1 = lr.conv2D_layer_bn(pool4, num_filters=1024)
    conv5_2 = lr.conv2D_layer_bn(conv5_1, num_filters=1024)
    upconv4 = lr.upsample(conv5_2)

    concat4 = layers.concatenate([conv4_2, upconv4], axis=-1)

    conv6_1 = lr.conv2D_layer_bn(concat4, num_filters=512)
    conv6_2 = lr.conv2D_layer_bn(conv6_1, num_filters=512)
    upconv3 = lr.upsample(conv6_2)
    
    concat3 = layers.concatenate([conv3_2, upconv3], axis=-1)

    conv7_1 = lr.conv2D_layer_bn(concat3, num_filters=256)
    conv7_2 = lr.conv2D_layer_bn(conv7_1, num_filters=256)
    upconv2 = lr.upsample(conv7_2)
    
    concat2 = layers.concatenate([conv2_2, upconv2], axis=-1)

    conv8_1 = lr.conv2D_layer_bn(concat2, num_filters=128)
    conv8_2 = lr.conv2D_layer_bn(conv8_1, num_filters=128)
    upconv1 = lr.upsample(conv8_2)
    
    concat1 = layers.concatenate([conv1_2, upconv1], axis=-1)

    conv9_1 = lr.conv2D_layer_bn(concat1, num_filters=64)
    conv9_2 = lr.conv2D_layer_bn(conv9_1, num_filters=64)
    
    if nlabels == 2:
        pred = lr.conv2D_layer_bn(conv9_2, num_filters=1, kernel_size=(1,1), activation='sigmoid')
    elif nlabels >= 3:
        pred = lr.conv2D_layer_bn(conv9_2, num_filters=nlabels, kernel_size=(1,1), activation='softmax')

    return Model(inputs, pred)




def get_model(images, nlabels, config):
    
    return config.model_handle(images, nlabels)
