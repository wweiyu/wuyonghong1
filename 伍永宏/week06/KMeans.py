# coding=utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def kmeans_detail(x, k):
    row = len(x)
    if k >= row:
        return x
    '质点'
    centers = [x[i] for i in range(k)]
    result = [[] for i in range(k)]
    stop = False
    while not stop:
        cur = [[] for i in range(k)]
        for index in range(row):
            var = x[index]
            min_index = -1
            min_dis = -1
            for i in range(k):
                dis = (var[0] - centers[i][0]) ** 2 + (var[1] - centers[i][1]) ** 2
                if min_dis < 0 or min_dis > dis:
                    min_dis = dis
                    min_index = i
            cur[min_index].append(var)
        if cur == result:
            stop = True
        else:
            result = cur
            '重新计算质点'
            centers = []
            for item in result:
                centers.append(np.mean(item, axis=0))
    return result


if __name__ == "__main__":
    x = [[0.0888, 0.5885],
         [0.1399, 0.8291],
         [0.0747, 0.4974],
         [0.0983, 0.5772],
         [0.1276, 0.5703],
         [0.1671, 0.5835],
         [0.1306, 0.5276],
         [0.1061, 0.5523],
         [0.2446, 0.4007],
         [0.1670, 0.4770],
         [0.2485, 0.4313],
         [0.1227, 0.4909],
         [0.1240, 0.5668],
         [0.1461, 0.5113],
         [0.2315, 0.3788],
         [0.0494, 0.5590],
         [0.1107, 0.4799],
         [0.1121, 0.5735],
         [0.1007, 0.6318],
         [0.2567, 0.4326],
         [0.1956, 0.4280]
         ]
    result = kmeans_detail(x, 3)

    '绘图'
    index = 0
    color = list('rgb')
    plt.figure(figsize=(10,8))
    plt.subplot(231)
    for item in result:
        x1 = [var[0] for var in item]
        y1 = [var[1] for var in item]
        plt.scatter(x1, y1, s=10, c=color[index], marker='o')
        index += 1
    plt.title("Self_Kmeans-Basketball Data")
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")
    plt.legend(["A", "B", "S"])

    plt.subplot(233)
    clf = KMeans(n_clusters=3)
    res = clf.fit_predict(x)
    x1 = [var[0] for var in x]
    y1 = [var[1] for var in x]
    plt.scatter(x1, y1, s=10, c=res, marker='x')
    plt.title("Buildin_Kmeans-Basketball Data")
    plt.xlabel("assists_per_minute")
    plt.ylabel("points_per_minute")
    plt.legend(["A", "B", "S"])

    path = "../../image/lenna.png"
    img = cv2.imread(path)

    data = img.reshape((-1,3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,10,1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret,labels,centers = cv2.kmeans(data,2,None,criteria,10,flags)
    centers = np.uint8(centers)
    dst2 = centers[labels.flatten()]
    dst2 = dst2.reshape(img.shape)
    dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)

    ret,labels,centers = cv2.kmeans(data,4,None,criteria,10,flags)
    centers = np.uint8(centers)
    dst4 = centers[labels.flatten()]
    dst4 = dst4.reshape(img.shape)
    dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    imgs = [img,dst2,dst4]
    titles = ["原始图像","聚类图像 K=2","聚类图像 K=4"]
    for i in range(3):
        plt.subplot(2,3,i+4),plt.imshow(imgs[i],'gray'),plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
