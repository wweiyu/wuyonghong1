from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch
import torchvision
import numpy as np
import cv2

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = maskrcnn_resnet50_fpn(pretrained = True)
    model.eval()
    model.to(device)

    path = '../imgs/street.jpg'
    draw_img = cv2.imread(path)
    draw_img = cv2.cvtColor(draw_img,cv2.COLOR_BGR2RGB)

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    image_tensor = trans(draw_img).unsqueeze(0)
    image_tensor.to(device)

    with torch.no_grad():
        prediction = model(image_tensor)
    clolors = {}
    for pred in prediction:
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        masks = pred['masks'].cpu().numpy()
        print(np.shape(masks),np.shape(scores),np.shape(labels))
        for i,(score,label,mask) in enumerate(zip(scores,labels,masks)):
            if score > 0.5:
                mask = mask[0]
                mask = (mask > 0.5).astype(np.uint8)
                if i not in clolors:
                    clolors[i] = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
                color = clolors[i]
                contours,_ = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(draw_img,contours,-1,color,2)

    cv2.imshow('reslut',draw_img)
    cv2.waitKey(0)
