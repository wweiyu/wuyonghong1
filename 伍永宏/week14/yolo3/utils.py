from PIL import Image
import numpy as np
import tensorflow as tf

def letterbox_image(image,size):
    ' 对预测输入图像进行缩放，按照长宽比进行缩放，不足的地方进行填充'
    image_w,image_h = image.size
    w,h = size
    scale = min(w*1.0/image_w,h*1.0/image_h)
    new_w = int(image_w * scale)
    new_h = int(image_h * scale)

    resize_image = image.resize((new_w,new_h),Image.BICUBIC)
    boxed_image = Image.new('RGB',size,(128,128,128))
    boxed_image.paste(resize_image,((w-new_w)//2,(h-new_h)//2))
    return boxed_image

def load_weights(var_list,weights_file):
    with open(weights_file, "rb") as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)

        weights = np.fromfile(fp, dtype=np.float32)
    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]
        if 'conv2d' in var1.name.split('/')[-2]:
            if 'bn' in var2.name.split('/')[-2]:
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                # we move the pointer by 4, because we loaded 4 variables
                i += 4
            elif 'conv2d' in var2.name.split('/')[-2]:
                # load biases
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                # we loaded 1 variable
                i += 1
            # we can load weights of conv layer
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # remember to transpose to column-major
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
            i += 1
    return assign_ops
