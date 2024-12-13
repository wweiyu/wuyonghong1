import tensorflow as tf
import tf_slim as slim

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):
    with tf.variable_scope(scope, 'vgg_16', [inputs]):

        # ' 2次卷积 224 * 224 * 64'
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')

        '[2x2]池化 112 * 112 * 64'
        net = slim.max_pool2d(net,[2,2],padding='VALID',scope = 'pool1')

        # '2次卷积 112 * 112 * 128'
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        '[2x2]池化 56 * 56 * 128'
        net = slim.max_pool2d(net, [2, 2], padding='VALID', scope='pool2')


        # '3次卷积 56 * 56 * 256'
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        '[2x2]次池化 28 * 28 * 256'
        net = slim.max_pool2d(net, [2, 2], padding='VALID', scope='pool3')

        # '3次卷积 28 * 28 512'
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        '[2x2]池化 14 * 14 * 512'
        net = slim.max_pool2d(net, [2, 2], padding='VALID', scope='pool4')


        # '3次卷积 14 * 14 512'
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        '[2x2]池化 7 * 7 * 512'
        net = slim.max_pool2d(net, [2, 2], padding='VALID', scope='pool5')

        # '全连接  1 * 1 * 4096'
        net = slim.conv2d(net,4096,[7,7],padding='VALID',scope='fc6')
        # 'dropout'
        net = slim.dropout(net,keep_prob=dropout_keep_prob,is_training = is_training,scope = 'dropout6')
        '全连接  1 * 1 * 4096'
        net = slim.conv2d(net,4096,[1,1],padding='VALID',scope='fc7')
        'dropout'
        net = slim.dropout(net,keep_prob=dropout_keep_prob,is_training = is_training,scope = 'dropout7')
        '全连接 1*1*1000'
        net = slim.conv2d(net,num_classes,[1,1],
                          activation_fn=None,
                          normalizer_fn=None, scope='fc8')

        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net

