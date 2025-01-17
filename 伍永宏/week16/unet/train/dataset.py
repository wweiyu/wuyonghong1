from torch.utils.data import Dataset
import glob
import os
import cv2
import random
import torch


class ISBI_Dataset(Dataset):
    def __init__(self,data_path):
        super().__init__()
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))

    def augment(self,image,flipCode):
        flip = cv2.flip(image,flipCode)
        return flip

    def __getitem__(self,index):
        img_path = self.imgs_path[index]
        label_path = img_path.replace('image','label')

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = img.reshape((1,img.shape[0],img.shape[1]))

        label = cv2.imread(label_path)
        label = cv2.cvtColor(label,cv2.COLOR_BGR2GRAY)
        label = label.reshape((1,label.shape[0],label.shape[1]))
        if label.max() > 1:
            label = label/255

        flipcode = random.choice([-1,0,1,2])
        if flipcode != 2:
            img = self.augment(img,flipcode)
            label = self.augment(label,flipcode)
        return img,label

    def __len__(self):
        return len(self.imgs_path)

if __name__ == "__main__":
    isbi_dataset = ISBI_Dataset("../data/train/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
