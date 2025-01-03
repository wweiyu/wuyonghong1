import tensorflow as tf
from PIL import Image,ImageFont,ImageDraw
import scripts.lesson14.yolo.utils as utils
import numpy as np
from scripts.lesson14.yolo.yolo_predict import yolo_predictor
import scripts.lesson14.yolo.config as config

if __name__ == "__main__":
    img_path = 'img/img.jpg'
    image = Image.open(img_path)
    resize_img = utils.letterbox_image(image, (416, 416))
    image_data = np.array(resize_img,dtype=np.float32)
    image_data = image_data/255.0
    image_data = np.expand_dims(image_data,axis=0)

    # image_input_shape = tf.placeholder(shape=(2,),dtype=tf.int32)
    input_image_shape = tf.placeholder(dtype = tf.int32, shape = (2,))
    inputs = tf.placeholder(shape=(None,416,416,3),dtype=tf.float32)

    predictor = yolo_predictor(config.obj_threshold, config.nms_threshold, config.anchors_path, config.classes_path)

    with tf.Session() as sess:
        with tf.variable_scope('predict'):
            boxes,scores,classes = predictor.predict(inputs,input_image_shape)
        load_op = utils.load_weights(tf.global_variables('predict'), config.weight_path)
        sess.run(load_op)

        out_boxes, out_scores, out_classes = sess.run([boxes,scores,classes],
                                                      feed_dict={inputs:image_data,
                                                                 input_image_shape: [image.size[1], image.size[0]]})

    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

    # font = ImageFont.truetype(size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    # 厚度
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        # 获得预测名字，box和分数
        predicted_class = predictor.class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        # 打印
        label = '{} {:.2f}'.format(predicted_class, score)

        # 用于画框框和文字
        draw = ImageDraw.Draw(image)
        # textsize用于获得写字的时候，按照这个字体，要多大的框
        label_size = draw.textsize(label, font)

        # 获得四个边
        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1] - 1, np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0] - 1, np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))
        print(label_size)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=predictor.colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=predictor.colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    image.show()
    image.save('./img/result1.jpg')
