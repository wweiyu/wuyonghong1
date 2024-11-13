import numpy as np
import cv2

'均值哈希'
def get_ahash(img):
    scl_img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scl_img,cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    gray[gray < mean] = 0
    gray[gray >= mean] = 1
    str = np.array2string(gray.flatten())
    return str[1:-1]

'差值哈希'
def get_dhash(img):
    scl_img = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scl_img,cv2.COLOR_BGR2GRAY)
    data = np.array([[1 if gray[x][y] > gray[x][y+1] else 0 for y in range(8)] for x in range(8)])
    str = np.array2string(data.flatten())
    return str[1:-1]

def compare_hash(a_hash,b_hash):
    if len(a_hash) != len(b_hash):
        return -1
    count = 0
    for i in range(len(a_hash)):
        if a_hash[i] == b_hash[i]:
            count += 1
    return count

if __name__ == "__main__":
    path = '../../image/iphone1.png'
    path2 =  '../../image/iphone2.png'
    img = cv2.imread(path)
    img2 = cv2.imread(path2)

    a_hash_1 = get_ahash(img)
    a_hash_2 = get_ahash(img2)
    print('均值哈希算法相似度：', compare_hash(a_hash_1,a_hash_2))

    d_hash_1 = get_dhash(img)
    d_hash_2 = get_dhash(img2)
    print('差值哈希算法相似度：', compare_hash(d_hash_1,d_hash_2))
