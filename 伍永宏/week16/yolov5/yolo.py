import torch
from yolov5.utils import *
import colorsys
from yolov5.model.yolov5 import YoloV5
from yolov5.utils_bbox import DecodeBox
import cv2
from PIL import Image, ImageDraw,ImageFont

class Yolo():
    _defaults = {
        'model_path':'./model_data/yolov5_s_v6.1.pth',
        'classes_path':'./model_data/coco_classes.txt',
        'anchors_path':'./model_data/yolo_anchors.txt',
        "anchors_mask": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "input_shape": [640, 640],
        "phi": 's',
        "confidence": 0.5,
        "nms_iou": 0.3,

    }
    def __init__(self):
       self.__dict__.update(self._defaults)
       self.class_names,self.class_nums = get_classes(self.classes_path)
       self.anchors, self.num_anchors = get_anchors(self.anchors_path)
       self.box_utils = DecodeBox(self.anchors,self.class_nums,self.input_shape,self.anchors_mask)

       hsv_tuples = [(x / self.class_nums, 1., 1.) for x in range(self.class_nums)]
       self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
       self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
       self.generate()

    def generate(self):
        self.yolo = YoloV5(self.anchors_mask,self.class_nums,self.phi,False)
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        # from torchsummary import summary
        # summary(self.yolo, (3, 640, 640))
        self.yolo.load_state_dict(torch.load(self.model_path,map_location = device))
        self.yolo.eval()

    def detect_image(self,image):
        image_copy = image.copy()
        image_shape = np.array(np.shape(image)[0:2])
        image = np.array(image.resize(self.input_shape),dtype=np.float64)
        image = image/255.
        image = np.array(image,dtype=np.float32)
        image = np.expand_dims(np.transpose(image,(2,0,1)),axis=0)
        print(image.shape)
        with torch.no_grad():
            image_tensor = torch.from_numpy(image)
            output = self.yolo(image_tensor)
            output = self.box_utils.decode_bbox(output)
            results = self.box_utils.non_max_suppression(torch.cat(output,1),image_shape,self.confidence,self.nms_iou)
            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image_shape[1] + 0.5).astype('int32'))
        thickness = int(max((image_shape[0] + image_shape[1]) // np.mean(self.input_shape), 1))
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image_shape[1], np.floor(bottom).astype('int32'))
            right = min(image_shape[0], np.floor(right).astype('int32'))

            label = '{}'.format(predicted_class)
            draw = ImageDraw.Draw(image_copy)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            font_color = (0, 0, 0)
            color = (0, 0, 255)
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=color)
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color)
            draw.text(text_origin, str(label, 'UTF-8'), fill=font_color, font=font)
            del draw
        image_copy.show()


if __name__ == "__main__":
    path = '../../../image/street.jpg'
    yolo = Yolo()
    image = Image.open(path)
    yolo.detect_image(image)
