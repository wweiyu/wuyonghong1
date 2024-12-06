import tensorflow.compat.v1 as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.nn as tfnn

tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":

    s1 = 200
    x_data = np.linspace(-1,1,s1)[:,np.newaxis]
    noise = np.random.normal(0,0.02,x_data.shape)
    y_data = np.square(x_data) + noise

    x = tf.placeholder(tf.float32, shape= x_data.shape)
    y = tf.placeholder(tf.float32,shape= y_data.shape)

    s2,s3 = 10,200
    w1 = tf.Variable(tf.random.normal([1,s2]))
    bias_1 = tf.Variable(tf.random.normal([1,s2]))
    in_put1 = tf.matmul(x,w1) + bias_1
    out_put1 = tfnn.tanh(in_put1)

    w2 = tf.Variable(tf.random.normal([s2,1]))
    bias_2 = tf.Variable(tf.random.normal([1,1]))
    in_put2 = tf.matmul(out_put1,w2) + bias_2
    out_put2 = tfnn.tanh(in_put2)

    loss = tf.reduce_mean(tf.square(out_put2 - y ))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        epoch = 2000
        for i in range(epoch):
            session.run(train_step,feed_dict={x:x_data,y:y_data})
        # res = session.run(out_put2, feed_dict={x: x_data})
        x_test = np.random.normal(0,0.5,size=(200,1))
        res = session.run(out_put2,feed_dict={x:x_test})
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.scatter(x_test,res,c ="r")
        # plt.plot(x_data,res,"r-",lw = 5)
        plt.show()
