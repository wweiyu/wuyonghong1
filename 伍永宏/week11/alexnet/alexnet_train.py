from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import cv2
import numpy as np
from alexnet.alexnet_model import alexnet_model
from keras.optimizers import Adam
import matplotlib.image as mpimg


def generate_data_from_file(lines,batch_size):
    i = 0
    num = len(lines)
    while True:
        x = []
        y = []
        for batch in range(batch_size):
            value = lines[i]
            arr = value.split(',')
            img_name = arr[0]
            img_path = f'./alexnet_train_data/train/{img_name}'
            try:
                img = mpimg.imread(img_path)
                min_len = min(img.shape[:2])
                hi, wd = img.shape[:2]
                y_y = (hi - min_len) // 2
                x_x = (wd - min_len) // 2
                img = img[y_y:y_y + min_len, x_x:x_x + min_len]
                img = cv2.resize(img, (224,224))
                img = np.float32(img/255)
                x.append(img)
                y.append(int(arr[1]))
            except:
                print('path error :',img_path)
            i = (i+1)%num
        x_train,y_train = np.array(x),np.array(y)
        x_train.reshape((-1,112,112,3))
        yield (x_train,y_train)



if __name__ == "__main__":

    model = alexnet_model()
    log_dir = './logs/'
    check_point = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                  monitor = 'acc',
                                  verbose = 1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto',
                                  period=3)

    reduce_lr = ReduceLROnPlateau(monitor ='val_loss',
                                  factor= 0.5,
                                  patience=3,
                                  verbose=0)

    early_call = EarlyStopping( monitor='val_loss',
                                min_delta=0,
                                patience= 10,
                                verbose= 0)


    path = './alexnet_train_data/dataset.txt'
    with open(path,'r') as ff:
        cur = ff.readlines()
        lines = cur[0].split('#')
    np.random.shuffle(lines)

    all_num = len(lines)
    train_num = int(all_num * 0.9 )
    test_num = all_num - train_num

    batch_size = 128
    epoch = 50

    # 交叉熵
    model.compile(loss = 'sparse_categorical_crossentropy',
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])

    model.fit_generator(generate_data_from_file(lines[:train_num],batch_size),
                        steps_per_epoch = train_num // batch_size,
                        epochs = epoch,
                        callbacks = [check_point,reduce_lr,early_call],
                        validation_data = generate_data_from_file(lines[train_num:],batch_size),
                        validation_steps = test_num // batch_size,
                        workers = 1)
    model.save_weights(log_dir + 'best.h5')

