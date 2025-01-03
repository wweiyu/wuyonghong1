from keras.layers import Input,Conv2D,MaxPooling2D,Flatten,Dense,Permute
from keras.layers.advanced_activations import  PReLU
from keras.models import Model
import scripts.lesson14.mscnn.utils as utils
import cv2
import numpy as np

def create_pnet(weight_path):
    inputs = Input(shape=(None,None,3))

    x = Conv2D(10,kernel_size=(3,3),strides=1,name='conv1')(inputs)
    x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)

    x = Conv2D(16,kernel_size=(3,3),strides=1,name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU2')(x)

    x = Conv2D(32,kernel_size=(3,3),strides=1,name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU3')(x)

    classifier = Conv2D(2,kernel_size=(1,1),activation='softmax',name='conv4_1')(x)

    bbox = Conv2D(4,kernel_size=(1,1),name='conv4_2')(x)

    model = Model([inputs],[classifier,bbox])
    model.load_weights(weight_path)
    return model

def create_rnet(weight_path):
    inputs = Input(shape=[24,24,3])
    '24,24,3 -> 11*11*28'
    x = Conv2D(28,kernel_size=(3,3),strides=1,name='conv1')(inputs)
    x = PReLU(shared_axes=[1,2],name='PRelu1')(x)
    x = MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)

    '11*11*28 -> 4*4*48'
    x = Conv2D(48, kernel_size=(3, 3), strides=1, name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='PRelu2')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2)(x)

    ' 4*4*48 -> 3*3*64'
    x = Conv2D(64, kernel_size=(2, 2), strides=1, name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PRelu3')(x)

    '3*3*64 -> 64*3*3'
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    '576->128'
    x = Dense(128,name='conv4')(x)
    x = PReLU(name='PRelu4')(x)

    classifier = Dense(2,activation='softmax',name='conv5_1')(x)
    bbox = Dense(4,name='conv5_2')(x)

    model = Model([inputs],[classifier,bbox])
    model.load_weights(weight_path)
    return model

def create_onet(weight_path):
    inputs = Input(shape=[48,48,3])
    '48,48,3 -> 23*23*32'
    x = Conv2D(32,kernel_size=(3,3),strides=1,name='conv1')(inputs)
    x = PReLU(shared_axes=[1,2],name='PRelu1')(x)
    x = MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)

    '23*23*32 -> 8*8*64'
    x = Conv2D(64,kernel_size=(3,3),strides=1,name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PRelu2')(x)
    x = MaxPooling2D(pool_size=(3,3),strides=2)(x)

    '8*8*64 -> 4*4*64'
    x = Conv2D(64, kernel_size=(3, 3), strides=1, name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PRelu3')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    '4*4*64 -> 3*3*128'
    x = Conv2D(128, kernel_size=(2, 2), strides=1, name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='PRelu4')(x)

    '3*3*128 -> 128*3*3'
    x = Permute([3,2,1])(x)
    x = Flatten()(x)

    '1152 -> 128'
    x = Dense(256,name='conv5')(x)
    x = PReLU(name='PRelu5')(x)

    classifier = Dense(2,activation='softmax',name='conv6_1')(x)
    bbox = Dense(4,name='conv6_2')(x)
    locations = Dense(10,name='conv6_3')(x)

    model = Model([inputs],[classifier,bbox,locations])
    model.load_weights(weight_path)
    return model

class mscnn():
    def __init__(self):
        self.p_net = create_pnet('./model_data/pnet.h5')
        self.r_net = create_rnet('./model_data/rnet.h5')
        self.o_net = create_onet('./model_data/onet.h5')

    def detect(self,image,threshold):
        scales = utils.calculateScales(image)
        copy_image = (image.copy() - 127.5)/127.5
        original_h,original_w = copy_image.shape[:2]

        out = []
        for scale in scales:
            cur_h,cur_w = int(original_h*scale),int(original_w*scale)
            cur_img = cv2.resize(copy_image,(cur_w,cur_h))
            inputs = cur_img.reshape(1,*cur_img.shape)
            output = self.p_net.predict(inputs)
            out.append(output)

        rectangles = []
        # print(np.array(out).shape)
        for i in range(len(scales)):
            prob = out[i][0][0][:,:,1]
            bbx = out[i][1][0]
            # print('out_shape =', prob.shape,bbx.shape,'scale = ',scales[i])

            cur_h,cur_w = prob.shape
            out_side = max(cur_h,cur_w)
            rect =  utils.detect_face_12net(prob,bbx,out_side,1/scales[i], original_w,original_h,threshold[0])
            rectangles.extend(rect)

        rectangles = utils.NMS(rectangles,0.7)
        if len(rectangles) == 0:
            return rectangles

        predict_24_batch = []
        for rectangle in rectangles:
            try:
                crop_img = copy_image[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                scale_img = cv2.resize(crop_img, (24, 24))
                predict_24_batch.append(scale_img)
            except:
                print('index=',int(rectangle[1]),int(rectangle[3]), int(rectangle[0]),int(rectangle[2]))

        predict_24_batch = np.array(predict_24_batch)
        out = self.r_net.predict(predict_24_batch)

        cls_prob = np.array(out[0])
        roi_prob = np.array(out[1])
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, original_w, original_h, threshold[1])

        if len(rectangles) == 0:
            return rectangles

        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_image[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv2.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.o_net.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]

        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, original_w, original_h, threshold[2])

        return rectangles
