import torch
from unet.model.unet import Unet
import cv2
import numpy as np
import glob

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(1,1)
    model.to(device)
    model.load_state_dict(torch.load('best_model.pth',map_location=device))
    model.eval()
    path = './data/test/0.png'
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(img_gray.shape)
    img_gray =img_gray.reshape((1,1,img_gray.shape[0],img_gray.shape[1]))
    img_tensor = torch.from_numpy(img_gray)
    img_tensor = img_tensor.to(device,dtype=torch.float32)
    result = model(img_tensor)
    result = np.array(result.data.cpu()[0])[0]
    result[result >= 0.5] = 255
    result[result <0.5] = 0
    cv2.imshow('result',result)
    cv2.waitKey(0)


