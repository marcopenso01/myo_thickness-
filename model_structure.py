import tensorflow as tf

from tfwrapper import layers

import logging


#same
def unet2D_same(images, training, nlabels):

    conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=64, training=training)
    logging.info('conv1_1')
    logging.info(conv1_1.shape)
    conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=64, training=training)
    logging.info('conv1_2')
    logging.info(conv1_2.shape)

    pool1 = layers.max_pool_layer2d(conv1_2)
    logging.info('pool1')
    logging.info(pool1.shape)

    conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=128, training=training)
    logging.info('conv2_1')
    logging.info(conv2_1.shape)
    conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=128, training=training)
    logging.info('conv2_2')
    logging.info(conv2_2.shape)
    
    pool2 = layers.max_pool_layer2d(conv2_2)
    logging.info('pool2')
    logging.info(pool2.shape)

    conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=256, training=training)
    logging.info('conv3_1')
    logging.info(conv3_1.shape)
    conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=256, training=training)
    logging.info('conv3_2')
    logging.info(conv3_2.shape)

    pool3 = layers.max_pool_layer2d(conv3_2)
    logging.info('pool3')
    logging.info(pool3.shape)

    conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=512, training=training)
    logging.info('conv4_1')
    logging.info(conv4_1.shape)
    conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=512, training=training)
    logging.info('conv4_2')
    logging.info(conv4_2.shape)

    pool4 = layers.max_pool_layer2d(conv4_2)
    logging.info('pool4')
    logging.info(pool4.shape)

    conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=1024, training=training)
    logging.info('conv5_1')
    logging.info(conv5_1.shape)
    conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=1024, training=training)
    logging.info('conv5_2')
    logging.info(conv5_2.shape)
    
    upconv4 = layers.upsample(conv5_2)
    #upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=512, weight_init='bilinear', training=training)
    logging.info('upconv4')
    logging.info(upconv4.shape)
    concat4 = layers.crop_and_concat_layer([conv4_2, upconv4], axis=3)
    logging.info('concat4')
    logging.info(concat4.shape)

    conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=512, training=training)
    logging.info('conv6_1')
    logging.info(conv6_1.shape)
    conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=512, training=training)
    logging.info('conv6_2')
    logging.info(conv6_2.shape)
    
    upconv3 = layers.upsample(conv6_2)
    #upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=256, weight_init='bilinear', training=training)
    logging.info('upconv3')
    logging.info(upconv3.shape)
    concat3 = tf.concat([conv3_2, upconv3], axis=3, name='concat3')
    logging.info('concat3')
    logging.info(concat3.shape)

    conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=256, training=training)
    logging.info('conv7_1')
    logging.info(conv7_1.shape)
    conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=256, training=training)
    logging.info('conv7_2')
    logging.info(conv7_2.shape)

    upconv2 = layers.upsample(conv7_2)
    #upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=128, weight_init='bilinear', training=training)
    logging.info('upconv2')
    logging.info(upconv2.shape)
    concat2 = tf.concat([conv2_2, upconv2], axis=3, name='concat2')
    logging.info('concat2')
    logging.info(concat2.shape)

    conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=128, training=training)
    logging.info('conv8_1')
    logging.info(conv8_1.shape)
    conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=128, training=training)
    logging.info('conv8_2')
    logging.info(conv8_2.shape)

    upconv1 = layers.upsample(conv8_2)
    #upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=64, weight_init='bilinear', training=training)
    logging.info('upconv1')
    logging.info(upconv1.shape)
    concat1 = tf.concat([conv1_2, upconv1], axis=3, name='concat1')
    logging.info('concat1')
    logging.info(concat1.shape)

    conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=64, training=training)
    logging.info('conv9_1')
    logging.info(conv9_1.shape)
    conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=64, training=training)
    logging.info('conv9_2')
    logging.info(conv9_2.shape)

    pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)
    logging.info('pred')
    logging.info(pred.shape)
    
    return pred


#ResUnet
'''
As described in https://arxiv.org/pdf/1711.10684.pdf
'''
def ResUNet(images, training, nlabels):
    
    #encoder
    e1 = layers.res_block_initial(images, 'e1', num_filters=64, strides=[1,1], training=training)
    e2 = layers.residual_block(e1, 'e2', num_filters=128, strides=[2,1], training=training)
    e3 = layers.residual_block(e2, 'e3', num_filters=256, strides=[2,1], training=training)
    
    #bridge layer, number of filters is double that of the last encoder layer
    b0 = layers.residual_block(e3, 'b0', num_filters=512, strides=[2,1], training=training)
    
    #decoder
    up3 = layers.upsample(b0)
    concat3 = tf.concat([up3, e3], axis=3,  name='concat3')
    d3 = layers.residual_block(concat3, 'd3', num_filters=256, strides=[1,1], training=training)
    
    up2 = layers.upsample(d3)
    concat2 = tf.concat([up2, e2], axis=3,  name='concat2')
    d2 = layers.residual_block(concat2, 'd2', num_filters=128, strides=[1,1], training=training)
    
    up1 = layers.upsample(d2)
    concat1 = tf.concat([up1, e1], axis=3,  name='concat1')
    d1 = layers.residual_block(concat1, 'd1', num_filters=64, strides=[1,1], training=training)
    
    pred = layers.conv2D_layer(d1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity)
    
    return pred
    

#Dense-UNet
'''
As described in https://arxiv.org/pdf/1709.07330.pdf
Downsampling: DenseNet-121
'''
def DenseUNet(images, training, nlabels):
    
    #encoder
    conv1 = layers.conv2D_layer_bn(images, 'conv1', num_filters=64, kernel_size=(7,7), strides=(2,2), training=training)
    
    pool1 = layers.max_pool_layer2d(conv1)

    dens1 = layers.dense_block(pool1, 'dens1', growth_rate=32, n_layers=6, training=training)
    trans1 = layers.transition_layer(dens1, 'trans1', num_filters=128, training=training)
    
    dens2 = layers.dense_block(trans1, 'dens2', growth_rate=32, n_layers=12, training=training)
    trans2 = layers.transition_layer(dens2, 'trans2', num_filters=256, training=training)
    
    dens3 = layers.dense_block(trans2, 'dens3', growth_rate=32, n_layers=24, training=training)
    trans3 = layers.transition_layer(dens3, 'trans3', num_filters=512, training=training)
    
    #bridge
    dens4 = layers.dense_block(trans3, 'dens4', growth_rate=32, n_layers=16, training=training)

    #decoder
    up1 = layers.upsample(dens4)
    concat1 = tf.concat([up1, dens3], axis=3, name='concat1')
    conv2 = layers.conv2D_layer_bn(concat1, 'conv2', num_filters=640, kernel_size=(3,3), training=training)
    
    up2 = layers.upsample(conv2)
    concat2 = tf.concat([up2, dens2], axis=3, name='concat2')
    conv3 = layers.conv2D_layer_bn(concat2, 'conv3', num_filters=256, kernel_size=(3,3), training=training)

    up3 = layers.upsample(conv3)
    concat3 = tf.concat([up3, dens1], axis=3, name='concat3')
    conv4 = layers.conv2D_layer_bn(concat3, 'conv4', num_filters=64, kernel_size=(3,3), training=training)
    
    up4 = layers.upsample(conv4)
    concat4 = tf.concat([up4, conv1], axis=3, name='concat4')
    conv5 = layers.conv2D_layer_bn(concat4, 'conv5', num_filters=64, kernel_size=(3,3), training=training)
    
    up5 = layers.upsample(conv5)
    conv6 = layers.conv2D_layer_bn(up5, 'conv6', num_filters=48, kernel_size=(3,3), training=training)
    
    pred = layers.conv2D_layer_bn(conv6, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)

    return pred
    

#Proposed network
def net1(images, training, nlabels):

    #encoder
    print('images', images.shape)
    select1 = layers.selective_kernel_block(images, 'select1', num_filters=32, training=training)
    print('select1', select1.shape)
    
    pool1 = layers.max_pool_layer2d(select1)
    print('pool1', pool1.shape)
    
    select2 = layers.selective_kernel_block(pool1, 'select2', num_filters=48, training=training)
    print('select2', select2.shape)
    dens2 = layers.dense_block(select2, 'dens2', growth_rate=16, n_layers=2, training=training)
    print('dens2', dens2.shape)
    
    trans2 = layers.transition_layer(dens2, 'trans2', num_filters=48, pool=1, training=training)
    print('trans2', trans2.shape)
    
    select3 = layers.selective_kernel_block(trans2, 'select3', num_filters=96, training=training)
    print('select3', select3.shape)
    dens3 = layers.dense_block(select3, 'dens3', growth_rate=16, n_layers=3, training=training)
    print('dens3', dens3.shape)
    
    trans3 = layers.transition_layer(dens3, 'trans3', num_filters=96, pool=1, training=training)
    print('trans3', trans3.shape)
    
    select4 = layers.selective_kernel_block(trans3, 'select4', num_filters=192, training=training)
    print('select4', select4.shape)
    dens4 = layers.dense_block(select4, 'dens4', growth_rate=16, n_layers=4, training=training)
    print('dens4', dens4.shape)
    
    trans4 = layers.transition_layer(dens4, 'trans4', num_filters=192, pool=1, training=training)
    print('trans4', trans4.shape)
    
    #bridge
    b1 = layers.conv2D_layer_bn(trans4, 'b1', num_filters=384, training=training)
    print('b1', b1.shape)
    cbam = layers.conv_block_att_module(b1, 'cbam')
    print('cbam', cbam.shape)
    b2 = layers.conv2D_layer_bn(cbam, 'b2', num_filters=384, training=training)
    print('b2', b2.shape)
    
    #decoder    
    up4 = layers.upsample(b2)
    print('up4', up4.shape)
    
    att4 = layers.spatial_attention(dens4, 'att4')
    print('att4', att4.shape)
    proj4 = layers.projection(att4, 'proj4', num_filters=192, training=training)
    print('proj4', proj4.shape)
    c4 = tf.concat([proj4, up4], axis=3, name='c4')
    print('c4', c4.shape)
    conv4= layers.conv2D_layer_bn(c4, 'conv4', num_filters=192, training=training)
    print('conv4', conv4.shape)
    
    up3 = layers.upsample(conv4)
    print('up3', up3.shape)
    
    att3 = layers.spatial_attention(dens3, 'att3')
    print('att3', att3.shape)
    proj3 = layers.projection(att3, 'proj3', num_filters=96, training=training)
    print('proj3', proj3.shape)
    c3 = tf.concat([proj3, up3], axis=3, name='c3')
    print('c3', c3.shape)
    conv3= layers.conv2D_layer_bn(c3, 'conv3', num_filters=96, training=training)
    print('conv3', conv3.shape)
    
    up2 = layers.upsample(conv3)
    print('up2', up2.shape)
    
    att2 = layers.spatial_attention(dens2, 'att2')
    print('att2', att2.shape)
    proj2 = layers.projection(att2, 'proj2', num_filters=48, training=training)
    print('proj2', proj2.shape)
    c2 = tf.concat([proj2, up2], axis=3, name='c2')
    print('c2', c2.shape)
    conv2= layers.conv2D_layer_bn(c2, 'conv2', num_filters=48, training=training)
    print('conv2', conv2.shape)
    
    up1 = layers.upsample(conv2)
    print('up1', up1.shape)
    
    att1 = layers.spatial_attention(select1, 'att1')
    print('att1', att1.shape)
    proj1 = layers.projection(att1, 'proj1', num_filters=32, training=training)
    print('proj1', proj1.shape)
    c1 = tf.concat([proj1, up1], axis=3, name='c1')
    print('c1', c1.shape)
    conv1= layers.conv2D_layer_bn(c1, 'conv1', num_filters=32, training=training)
    print('conv1', conv1.shape)
    
    #deep supervision
    up3d = layers.upsample(conv4)
    print('up3d', up3d.shape)
    conv3d = layers.conv2D_layer_bn(up3d, 'conv3d', num_filters=96, kernel_size=(1,1), training=training)
    print('conv3d', conv3d.shape)
    
    sum3 = tf.add(conv3d, conv3)
    print('sum3', sum3.shape)
    
    up2d = layers.upsample(sum3)
    print('up2d', up2d.shape)
    conv2d = layers.conv2D_layer_bn(up2d, 'conv2d', num_filters=48, kernel_size=(1,1), training=training)
    print('conv2d', conv2d.shape)
    
    sum2 = tf.add(conv2d, conv2)
    print('sum2', sum2.shape)
    
    up1d = layers.upsample(sum2)
    print('up1d', up1d.shape)
    conv1d = layers.conv2D_layer_bn(up1d, 'conv1d', num_filters=32, kernel_size=(1,1), training=training)
    print('conv1d', conv1d.shape)
    
    sum1 = tf.add(conv1d, conv1)
    print('sum1', sum1.shape)

    pred = layers.conv2D_layer_bn(sum1, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)
    print('pred', pred.shape)
    
    return pred
