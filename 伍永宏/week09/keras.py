from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import numpy as np

if __name__ == "__main__":
    (x_ims,x_labels),(y_ims, y_labels) = mnist.load_data()
    '训练集'
    x_shape = x_ims.shape
    x_ims = x_ims.reshape((x_shape[0],-1))
    x_ims = np.float32(x_ims/255)

    '验证集'
    y_shape = y_ims.shape
    y_ims = y_ims.reshape((y_shape[0],-1))

    '测试集'
    p_ims = y_ims[50:500]
    t_labels = y_labels[50:500]

    y_ims = np.float32(y_ims / 255)
    # p_ims = y_ims[50:60]

    network = models.Sequential()
    # network.add(layers.Dense(512,activation='relu',input_shape=(x_ims.shape[1],)))
    # network.add(layers.Dense(10,activation='softmax'))

    network.add(layers.Dense(375,activation='sigmoid',input_shape=(x_ims.shape[1],)))
    network.add(layers.Dense(150,activation='sigmoid'))
    network.add(layers.Dense(10,activation='softmax'))

    network.compile(loss="categorical_crossentropy",metrics=['accuracy'],optimizer='rmsprop')

    x_labels = to_categorical(x_labels)
    y_labels = to_categorical(y_labels)

    network.evaluate(y_ims, y_labels, verbose=1)

    network.fit(x_ims,x_labels,epochs=5,batch_size=128)

    res = network.predict(p_ims)

    res_labels = np.full(len(p_ims),-1)
    # print(res,type(res_labels))
    for index in range(len(p_ims)):
        res_labels[index]  = np.argmax(res[index])
        # for i in range(len(arr)):
            # if arr[i] == 1:
            #     res_labels[index] = i
            #     break
    print("不相等的个数：",np.sum(t_labels != res_labels))
