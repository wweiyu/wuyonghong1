import tensorflow as tf
import numpy as np
import vgg16
import matplotlib.image as mpimg


def crop_image(path):
    img = mpimg.imread(path)
    min_len = min(img.shape[:2])
    hi, wd = img.shape[:2]
    y = (hi - min_len) // 2
    x = (wd - min_len) // 2
    return img[y:y + min_len, x:x + min_len]


def tf_image_resize_images(img,size,method=tf.image.ResizeMethod.BILINEAR,align_corners=False):
    with tf.name_scope('resize_image'):
        image = tf.expand_dims(img,0)
        image= tf.image.resize_images(image, (224, 224), method=method, align_corners=align_corners)
        image = tf.reshape(image, tf.stack([-1,size[0], size[1], 3]))
        return image

if __name__ == "__main__":

    tests = ['dog.jpg', 'table.jpg']

    path = f'./test_data/{tests[0]}'
    img = crop_image(path)

    inputs = tf.placeholder(tf.float32, shape=[None, None, 3])
    images = tf_image_resize_images(inputs,(224,224))

    prediction = vgg16.vgg_16(images)
    kpt_path = './model/vgg_16.ckpt'
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess,kpt_path)
        last = tf.nn.softmax(prediction)

        res = sess.run(last,feed_dict={inputs:img})
        print(np.argmax(res,axis=1))

        with open('synset.txt','r') as ff:
            lines = ff.readlines()
            sort_index = np.argsort(res[0])[::-1]
            print('top1  :',lines[sort_index[0]],res[0,sort_index[0]])
            top5 = [(lines[sort_index[i]],res[0,sort_index[i]]) for i in range(5)]
            print('top5 ',top5)
