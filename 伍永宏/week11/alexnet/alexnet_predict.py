import numpy as np
import cv2
from alexnet.alexnet_model import alexnet_model

if __name__ == "__main__":
    cur_dir = './alexnet_train_data/test/'
    imgs =[]
    for i in range(4):
        if i == 0:
            path = f'{cur_dir}/test.jpg'
        else:
            path = f'{cur_dir}/test{i+1}.jpg'
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_nor = np.float32(img/255)
        input_shape = (224,224)
        img_in = cv2.resize(img_nor,input_shape)
        imgs.append(img_in)

    imgs = np.array(imgs)

    model = alexnet_model()
    h5_path = './logs/last1.h5'
    model.load_weights(h5_path)
    res = model.predict(imgs)
    ans = np.array(['猫','狗'])
    print('result :',ans[np.argmax(res,axis=1)])

    h5_path = './logs/best.h5'
    model.load_weights(h5_path)
    res = model.predict(imgs)
    print('result :',ans[np.argmax(res,axis=1)])




