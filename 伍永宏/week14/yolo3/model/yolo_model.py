import tensorflow as tf

class yolo3_model():
    def __init__(self, norm_epsilon, norm_decay):
        '''
        :param norm_epsilon:  方差加上极小的数，防止除以0的情况
        :param norm_decay:  在预测时计算moving average时的衰减率
        '''
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay

    def _batch_normalization_layer(self,inputs,name = None,training = True,norm_decay = 0.99, norm_epsilon = 1e-3):
        bn_layer = tf.layers.batch_normalization(inputs = inputs,momentum = norm_decay, epsilon = norm_epsilon, center = True,
            scale = True, training = training, name = name)
        return tf.nn.leaky_relu(bn_layer, alpha = 0.1)

    def _conv2d_layer(self,inputs,filters,kernel_size,name,use_bias = False, strides = 1):
        # kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 5e-4),
        conv_layer = tf.layers.conv2d(inputs,filters,kernel_size,strides=[strides,strides],
                                      padding= 'same' if strides == 1 else 'valid',
                                      kernel_initializer= tf.glorot_uniform_initializer,
                                      kernel_regularizer = tf.keras.regularizers.l2(5e-4),
                                      name= name,use_bias=use_bias)

        return conv_layer

    def _residual_block(self,inputs,filters,blocks_num,conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        inputs = tf.pad(inputs,paddings=[[0,0],[1,0],[1,0],[0,0]], mode="CONSTANT")
        layer  = self._conv2d_layer(inputs,filters,kernel_size=3,strides=2,name='conv2d_{}'.format(conv_index))
        layer = self._batch_normalization_layer(layer,name='bn_{}'.format(conv_index),
                                                  training=training,norm_epsilon=norm_epsilon,norm_decay=norm_decay)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            '1x1'
            layer = self._conv2d_layer(layer,filters//2,kernel_size=1,name='conv2d_{}'.format(conv_index))
            layer = self._batch_normalization_layer(layer,name='bn_{}'.format(conv_index),
                                                  training=training,norm_epsilon=norm_epsilon,norm_decay=norm_decay)
            conv_index += 1
            '3x3'
            layer = self._conv2d_layer(layer, filters, kernel_size=3, name='conv2d_{}'.format(conv_index))
            layer = self._batch_normalization_layer(layer, name='bn_{}'.format(conv_index),
                                                    training=training, norm_epsilon=norm_epsilon, norm_decay=norm_decay)
            conv_index += 1
            layer = layer + shortcut
        return layer,conv_index

    def _darknet53(self,inputs,conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
            with tf.variable_scope('darknet53'):
                '416*416*3 -> 416*416*32'
                layer = self._conv2d_layer(inputs,32,kernel_size=3,name='conv2d_{}'.format(conv_index))
                layer = self._batch_normalization_layer(layer, name='bn_{}'.format(conv_index),
                                                        training=training, norm_epsilon=norm_epsilon,
                                                        norm_decay=norm_decay)
                conv_index += 1
                '416*416*32 -> 208*208*64'
                layer,conv_index = self._residual_block(layer,64,1,conv_index=conv_index,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)

                '208*208*64 -> 104*104*128'
                layer,conv_index = self._residual_block(layer,128,2,conv_index,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)

                '104*104*128 -> 52*52*256'
                layer,conv_index = self._residual_block(layer,256,8,conv_index,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
                out1 = layer
                '52*52*256 -> 26*26*512'
                layer,conv_index = self._residual_block(layer,512,8,conv_index,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)
                out2 = layer
                '26*26*512 -> 13*13*1024'
                layer,conv_index = self._residual_block(layer,1024,4,conv_index,training=training,norm_decay=norm_decay,norm_epsilon=norm_epsilon)

                return out1,out2,layer,conv_index

    '5L 1*1 3*3  1*1  3*3 1*1'
    '输出层再加2层  3*3  1*1'
    def _yolo_block(self,inputs,filters_num, out_filters,conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        '1*1'
        layer = self._conv2d_layer(inputs, filters_num, kernel_size=1, name='conv2d_{}'.format(conv_index))
        layer = self._batch_normalization_layer(layer, name='bn_{}'.format(conv_index),
                                                        training=training, norm_epsilon=norm_epsilon,
                                                        norm_decay=norm_decay)
        conv_index += 1

        '3*3'
        layer = self._conv2d_layer(layer, filters_num*2, kernel_size=3, name='conv2d_{}'.format(conv_index))
        layer = self._batch_normalization_layer(layer, name='bn_{}'.format(conv_index),
                                                training=training, norm_epsilon=norm_epsilon,
                                                norm_decay=norm_decay)
        conv_index += 1

        '1*1'
        layer = self._conv2d_layer(layer, filters_num, kernel_size=1, name='conv2d_{}'.format(conv_index))
        layer = self._batch_normalization_layer(layer, name='bn_{}'.format(conv_index),
                                                training=training, norm_epsilon=norm_epsilon,
                                                norm_decay=norm_decay)
        conv_index += 1

        '3*3'
        layer = self._conv2d_layer(layer, filters_num*2, kernel_size=3, name='conv2d_{}'.format(conv_index))
        layer = self._batch_normalization_layer(layer, name='bn_{}'.format(conv_index),
                                                training=training, norm_epsilon=norm_epsilon,
                                                norm_decay=norm_decay)
        conv_index += 1

        '1*1'
        layer = self._conv2d_layer(layer, filters_num, kernel_size=1, name='conv2d_{}'.format(conv_index))
        layer = self._batch_normalization_layer(layer, name='bn_{}'.format(conv_index),
                                                training=training, norm_epsilon=norm_epsilon,
                                                norm_decay=norm_decay)
        out1 = layer
        conv_index += 1

        '3*3'
        layer = self._conv2d_layer(layer, filters_num*2, kernel_size=3, name='conv2d_{}'.format(conv_index))
        layer = self._batch_normalization_layer(layer, name='bn_{}'.format(conv_index),
                                                training=training, norm_epsilon=norm_epsilon,
                                                norm_decay=norm_decay)
        conv_index += 1

        '1*1'
        layer = self._conv2d_layer(layer, out_filters, kernel_size=1, name='conv2d_{}'.format(conv_index),use_bias=True)
        conv_index += 1
        return out1,layer,conv_index

    def yolo_inference(self, inputs, num_anchors, num_classes, training = True):
        conv_index = 1
        # route1 = 52,52,256、route2 = 26,26,512、route3 = 13,13,1024
        conv52,conv26,conv13,conv_index = self._darknet53(inputs,conv_index,training,self.norm_decay,self.norm_epsilon)
        out_filters = num_anchors*(num_classes+5)
        with tf.variable_scope('yolo'):
            '第一个特征 13*13*num_anchors*(num_classes+5)'
            out1_1,out1,conv_index = self._yolo_block(conv13, 512,out_filters,conv_index,training,self.norm_decay,self.norm_epsilon)

            '第二个特征 26*26*num_anchors*(num_classes+5)'
            conv = self._conv2d_layer(out1_1,256,kernel_size=1,name='conv2d_{}'.format(conv_index))
            conv = self._batch_normalization_layer(conv,name='bn_{}'.format(conv_index),training=training,
                                                   norm_epsilon=self.norm_epsilon,norm_decay=self.norm_decay)
            conv_index += 1
            '上采样 26*26*256'
            unSample_0 = tf.image.resize_nearest_neighbor(conv, [2 * tf.shape(conv)[1], 2 * tf.shape(conv)[1]], name='upSample_0')
            route0 = tf.concat([unSample_0,conv26],axis=-1,name= 'route_0')

            out2_1,out2,conv_index = self._yolo_block(route0,256,out_filters,conv_index,training,self.norm_decay,self.norm_epsilon)

            '第三个特征 52*52*num_anchors*(num_classes+5)'
            conv = self._conv2d_layer(out2_1, 128, kernel_size=1, name='conv2d_{}'.format(conv_index))
            conv = self._batch_normalization_layer(conv, name='bn_{}'.format(conv_index), training=training,
                                                   norm_epsilon=self.norm_epsilon, norm_decay=self.norm_decay)
            conv_index += 1
            '上采样 52*52*128'
            unSample_1 = tf.image.resize_nearest_neighbor(conv,[tf.shape(conv)[1]*2,tf.shape(conv)[1]*2], name='upSample_1')
            'concat'
            route_1 = tf.concat([unSample_1,conv52], axis=-1, name='route_1')
            _,out3,_ = self._yolo_block(route_1,128,out_filters,conv_index,training,self.norm_decay,self.norm_epsilon)
            return [out1,out2,out3]

# model = Yolo3(config.norm_epsilon,config.norm_decay,config.anchors_path,config.classes_path,config.pre_train)
