import tensorflow.compat.v1 as tf
import numpy as np
import tensorflow.nn as tfnn
import os

tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    s1,s2,s3,s4 = 784,100,300,10

    input_d = tf.placeholder(tf.float32,shape=[s1,1])
    label = tf.placeholder(tf.float32,shape=[s4,1])

    w1 = tf.Variable(tf.random_normal([s2,s1]))
    bias_1 = tf.Variable(tf.random_normal([1,1]))
    input_1 = tf.matmul(w1,input_d) + bias_1
    output_1 = tfnn.sigmoid(input_1)

    # w2 = tf.Variable(tf.random.normal([s3,s2]))
    # bias_2 = tf.Variable(tf.random.normal([1,1]))
    # input_2 = tf.matmul(w2,output_1) + bias_2
    # output_2 = tfnn.sigmoid(input_2)

    # w3 = tf.Variable(tf.random.normal([s4,s3]))
    # input_3 = tf.matmul(w3,output_2) + bias_2
    # prediction = tfnn.softmax(input_3)

    w3 = tf.Variable(tf.random_normal([s4,s2]))
    input_3 = tf.matmul(w3,output_1)
    prediction = tfnn.sigmoid(input_3)

    loss = tf.reduce_mean(tf.square(prediction - label))
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        with open("../../../dataset/mnist_train.csv") as ff:
            res = ff.readlines()
            epoch = 50
            for i in range(epoch):
                for record in res:
                    data = record.split(',')
                    input_data = np.asfarray(data[1:],dtype=np.float32)/255.0
                    input_data = input_data[:,np.newaxis]
                    targets = np.zeros((s4,1),dtype=np.float32)
                    targets[int(data[0]),0] = 1
                    session.run(train_step,feed_dict={input_d:input_data,label:targets})

        with open("../../../dataset/mnist_test.csv") as tt:
            res = tt.readlines()
            total = len(res)
            accuracy = 0
            for record in res:
                data = record.split(',')
                input_data = np.asfarray(data[1:],dtype=np.float32) / 255.0
                input_data = input_data[:, np.newaxis]
                label = int(data[0])
                res = session.run(prediction, feed_dict={input_d: input_data})
                print('label = ',label,'res = ',np.argmax(res))
                if label == np.argmax(res):
                    accuracy += 1

            print("accuracy : ", accuracy / total)
