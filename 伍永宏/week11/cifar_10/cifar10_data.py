# import tensorflow.compat.v1 as tf
import tensorflow as tf
import os

#设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

class Cifar10_Record():
    pass


def get_cifar10_record(file_queue):
    label_bytes = 1
    result = Cifar10_Record()
    result.height = 32
    result.width = 32
    result.channel = 3
    image_bytes = result.width * result.height * result.channel
    all_bytes = image_bytes + label_bytes

    reader = tf.FixedLengthRecordReader(all_bytes)
    result.key,values = reader.read(file_queue)

    record_bytes = tf.decode_raw(values,tf.uint8)

    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)
    images = tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[all_bytes]),[result.channel,result.height,result.width])

    result.ui8_images = tf.transpose(images,[1,2,0])

    return result

def inputs_cifar10_data(pathDir,batch_size,disorder):
    files = [os.path.join(pathDir,f"data_batch_{i}.bin") for i in range(1,6)]
    file_queue = tf.train.string_input_producer(files)
    input_record = get_cifar10_record(file_queue)

    input_img = tf.cast(input_record.ui8_images,tf.float32)

    after_img = tf.random_crop(input_img, [24, 24, 3])

    min_after_dequeue = int(num_examples_pre_epoch_for_eval * 0.4)

    if disorder:
        '左右翻转'
        after_img = tf.image.flip_left_right(after_img)
        '明亮度'
        after_img = tf.image.random_brightness(after_img,max_delta = 0.75)
        '对比度'
        after_img = tf.image.random_contrast(after_img,lower=0.2,upper= 1.8)
        '归一化'
        after_img = tf.image.per_image_standardization(after_img)
        after_img.set_shape([24,24,3])                      #设置图片数据及标签的形状
        input_record.label.set_shape([1])

        image_batch,label_batch = tf.train.shuffle_batch([after_img,input_record.label],batch_size=batch_size,
                                                         capacity= min_after_dequeue + batch_size*3,
                                                         min_after_dequeue= min_after_dequeue,
                                                         num_threads = 16
                                                         )
        return image_batch,tf.reshape(label_batch,[batch_size])
    else:
        after_img = tf.image.per_image_standardization(after_img)
        after_img.set_shape([24, 24, 3])  # 设置图片数据及标签的形状
        input_record.label.set_shape([1])

        image_batch, label_batch = tf.train.batch([after_img,input_record.label],batch_size=batch_size,num_threads = 16,
                                                  capacity=min_after_dequeue + batch_size*3)

        return image_batch,tf.reshape(label_batch,[batch_size])
