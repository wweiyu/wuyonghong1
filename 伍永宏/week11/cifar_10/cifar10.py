import tensorflow.compat.v1 as tf
import cifar_10.cifar10_data as cifar10_data
import numpy as np

def variable_with_weight_loss(shape,stddev,w1):
    weights = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(weights),w1)
        tf.add_to_collection("losses",weight_loss)
    return weights

if __name__ == "__main__":
    path = './cifar-10-batches-bin/'

    batch_size = 128
    mat_step = 4000
    num_evl = 1000

    train_imgs,train_labels =  cifar10_data.inputs_cifar10_data(path,batch_size,True)
    test_imgs,test_labels =  cifar10_data.inputs_cifar10_data(path,batch_size,False)

    x = tf.placeholder(tf.float32,shape=[batch_size,24,24,3])
    y = tf.placeholder(tf.int32,shape=[batch_size])

    'conv2d_1'
    kernel_1 = variable_with_weight_loss([5,5,3,64],stddev=1e-5,w1=0.0)
    conv_1 = tf.nn.conv2d(x,kernel_1,strides=[1,1,1,1],padding='SAME')
    bias_1 = tf.Variable(tf.constant(0.1,shape=[64]))
    relu_1 = tf.nn.relu(tf.nn.bias_add(conv_1,bias_1))
    pool_1 = tf.nn.max_pool(relu_1,[1,3,3,1],strides=[1,1,1,1],padding='SAME')

    'conv2d_2'
    kernel_2 = variable_with_weight_loss([5,5,64,64],stddev=1e-5,w1=0.0)
    conv_2 = tf.nn.conv2d(pool_1,kernel_2,strides=[1,1,1,1],padding='SAME')
    bias_2 = tf.Variable(tf.constant(0.0,shape=[64]) )
    relu_2 = tf.nn.relu(tf.nn.bias_add(conv_2,bias_2))
    pool_2 = tf.nn.max_pool(relu_2,[1,3,3,1],strides=[1,1,1,1],padding='SAME')

    '全连接'
    pc_input = tf.reshape(pool_2,[batch_size,-1])
    dim = pc_input.get_shape()[1].value

    'fc_1'
    weight_1 = tf.Variable(variable_with_weight_loss([dim,382],stddev= 1e-3,w1=0.04))
    bias_3 = tf.Variable(tf.constant(0.1,shape=[382]))
    fc_1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(pc_input,weight_1),bias_3))

    'fc_2'
    weight_2 = tf.Variable(variable_with_weight_loss([382, 382], stddev=1e-3, w1=0.04))
    bias_4 = tf.Variable(tf.constant(0.01, shape=[382]))
    fc_2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc_1, weight_2), bias_4))

    'fc_3'
    weight_3 = tf.Variable(variable_with_weight_loss([382, 10], stddev=1e-3, w1=0.04))
    bias_5 = tf.Variable(tf.constant(0.01, shape=[10]))
    prediction = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc_2, weight_3), bias_5))

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,labels=y)

    weight_add_l2_loss = tf.add_n(tf.get_collection('losses'))
    loss = tf.reduce_mean(cross_entropy) + weight_add_l2_loss

    top_k = tf.nn.in_top_k(prediction,y,1)
    train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners()
        image_batch, label_batch = sess.run([train_imgs, train_labels])
        for step in range(mat_step):
            image_batch,label_batch = sess.run([train_imgs,train_labels])
            _,loss_value = sess.run([train_op,loss],feed_dict={x:image_batch,y:label_batch})
            if step % 100 == 0:
                print(f" step = {step},loss = {loss_value}")

        test_batch = int(num_evl/batch_size)
        count = 0
        for batch in range(test_batch):
            image_batch,label_batch = sess.run([test_imgs,test_labels])
            p = sess.run(top_k,feed_dict={x:image_batch,y:label_batch})
            count += np.sum(p)
        print('accuracy: ',count/(test_batch*batch_size))
