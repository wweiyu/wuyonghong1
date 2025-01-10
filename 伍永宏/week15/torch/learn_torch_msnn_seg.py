import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import numpy as np
import cv2

if __name__ == "__main__":

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    path = '../imgs/street.jpg'
    image = cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    model = maskrcnn_resnet50_fpn(pretrained = True)
    model.eval()
    model.to(device)

    image_tensor = torch.tensor(image)
    image_tensor = image_tensor/255.0
    image_tensor = torch.permute(image_tensor,(2,0,1))
    image_tensor = torch.unsqueeze(image_tensor,0)
    image_tensor.to(device)

    colors = {
        1: (255, 0, 0),  # 人用蓝色表示
        2: (0, 255, 0),  # 自行车用绿色表示
        3: (0, 0, 255)}  # 汽车用红色表示
    with torch.no_grad():
        prediction = model(image_tensor)
        for pred in prediction:
            masks = pred['masks'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            for mask,score,label in zip(masks,scores,labels):
                if score > 0.5:
                    mask = mask[0]
                    mask = (mask > 0.5).astype(np.uint8)
                    contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    color = colors.get(label.item(), (255, 255, 255))
                    cv2.drawContours(image,contours,-1,color,2)

    image = cv2.resize(image, (700, 700))
    cv2.imshow('Result', image)
    cv2.waitKey(0)
