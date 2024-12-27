import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image,ImageDraw

# print('torch.__version__ =',torch.__version__)
# print('torchvision.__version__ =',torchvision.__version__)
# print(torch.cuda.is_available())

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = fasterrcnn_resnet50_fpn(pretrained = True)
    model.eval()
    model = model.to(device)

    # img_path ='street.jpg'
    img_path ='bb.jpg'
    img = Image.open(img_path).convert('RGB')

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    inputs = trans(img).unsqueeze(dim=0)
    # print(inputs.size())
    inputs.to(device)
    #
    with torch.no_grad():
        prediction = model(inputs)

    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    draw = ImageDraw.Draw(img)
    for box,label,score in zip(boxes,labels,scores):
        if score > 0.5:  # 阈值可根据需要调整
            top_left = (box[0], box[1])
            bottom_right = (box[2], box[3])
            draw.rectangle([top_left, bottom_right], outline='red', width=2)
            draw.text((box[0], box[1] - 10), str(label), fill='green')
    img.show()
