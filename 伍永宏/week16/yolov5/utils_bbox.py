import numpy as np
import torch
from torchvision.ops import nms

class DecodeBox():
    def __init__(self,anchors,num_class,input_shape,anchor_mask):
        self.anchors = anchors
        self.num_classes = num_class
        self.bbox_attrs = 5 + num_class
        self.input_shape = input_shape
        self.anchors_mask = anchor_mask

    def decode_bbox(self,features):
        results = []
        for i,out in enumerate(features):
            batch_size = out.size(0)
            height = out.size(2)
            width = out.size(3)

            '640 / 20  40  80 = 32,16,4'
            stride_h = self.input_shape[0] / height
            stride_w = self.input_shape[1] / width

            ' 此时获得的scaled_anchors大小是相对于特征层的'
            scaled_anchors = [(a_w/stride_w,a_h/stride_h) for a_w,a_h in self.anchors[self.anchors_mask[i]]]

            'n,(5+num_class)*len(self.anchors_mask[i]),h,w-> n,len(self.anchors_mask[i]),h,w,5+num_class'
            prediction = out.view(batch_size,len(self.anchors_mask[i]),5+self.num_classes,height,width).permute(0,1,3,4,2).contiguous()

            x = torch.sigmoid(prediction[...,0])
            y = torch.sigmoid(prediction[...,1])
            w = torch.sigmoid(prediction[...,2])
            h = torch.sigmoid(prediction[...,3])

            label = torch.sigmoid(prediction[...,4])
            preds = torch.sigmoid(prediction[...,5:])

            grid_x = torch.linspace(0,width-1,width).repeat(height,1)
            grid_x = grid_x.repeat(batch_size*len(self.anchors_mask[i]),1,1).view(x.shape).type(torch.FloatTensor)

            grid_y = torch.linspace(0,height-1,height).repeat(width,1).t()
            grid_y = grid_y.repeat(batch_size*len(self.anchors_mask[i]),1,1).view(y.shape).type(torch.FloatTensor)

            anchor_w = torch.FloatTensor(scaled_anchors).index_select(1,torch.tensor(0))
            anchor_w = anchor_w.repeat(batch_size,1).repeat(1,1,width*height).view(w.shape)
            anchor_h = torch.FloatTensor(scaled_anchors).index_select(1,torch.tensor(1))
            anchor_h = anchor_h.repeat(batch_size,1).repeat(1,1,width*height).view(h.shape)

            pred_boxes = torch.FloatTensor(prediction[...,:4].shape)
            pred_boxes[...,0] = x.data * 2 - 0.5 + grid_x
            pred_boxes[...,1] = y.data * 2 - 0.5 + grid_y
            pred_boxes[...,2] = (w.data *2) ** 2 * anchor_w
            pred_boxes[...,3] = (h.data *2) ** 2 * anchor_h

            _scale = torch.Tensor([width,height,width,height]).type(torch.FloatTensor)
            out = torch.concat([pred_boxes.view(batch_size,-1,4)/_scale,label.view(batch_size,-1,1),
                                   preds.view(batch_size,-1,self.num_classes)],dim=-1)
            results.append(out.data)
        return results

    def yolo_correct_boxes(self,box_xy, box_wh, image_shape):
        box_yx = box_xy[...,::-1]
        box_hw = box_wh[...,::-1]
        input_shape = np.array(self.input_shape)
        image_shape = np.array(image_shape)

        box_min = box_yx - box_hw*0.5
        box_max = box_yx + box_hw*0.5
        # print('box_min',np.shape(box_min))
        boxes = np.concatenate([box_min[...,0:1],box_min[...,1:2],box_max[...,0:1],box_max[...,1:2]],axis=-1)
        # print('boxes.shape = ',boxes.shape)
        boxes *= np.concatenate([image_shape,image_shape],axis=-1)
        return boxes


    def non_max_suppression(self, prediction, image_shape, conf_thres=0.5, nms_thres=0.4):
        box_corner          = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        outputs = [None for _ in range(len(prediction))]
        for i,image_pred in enumerate(prediction):
            class_conf, class_pred = torch.max(image_pred[:,5:],1,keepdim=True)
            print('class_conf = ',class_conf.size())
            conf_mask = (image_pred[:,4] * class_conf[:,0] > conf_thres).squeeze()

            print('conf_mask = ',conf_mask.size())

            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue

            detection = torch.cat([image_pred[:,:5],class_conf.float(),class_pred.float()],dim=1)
            labels = detection[:,-1].cpu().unique()
            for c in labels:
                detections_class = detection[detection[:,-1] == c]
                keep = nms(detections_class[:,:4],detections_class[:,4] * detections_class[:,5],nms_thres)
                max_detections = detections_class[keep]
                outputs[i] = max_detections if outputs[i] is None else torch.cat([outputs[i],max_detections],dim=0)

            if outputs[i] is not None:
                outputs[i] = outputs[i].cpu().numpy()
                box_xy,box_wh = (outputs[i][:,:2] + outputs[i][:,2:4])/2,outputs[i][:,2:4] - outputs[i][:,:2]
                outputs[i][:,:4] = self.yolo_correct_boxes(box_xy,box_wh,image_shape)
        return outputs
