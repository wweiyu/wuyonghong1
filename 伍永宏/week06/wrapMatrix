import cv2
import numpy as np
import matplotlib.pyplot as plt

def getWrapMatrix(src,dst):
    assert src.shape == dst.shape and src.shape[0] >= 4
    num = src.shape[0]
    a_matrix = [None for i in  range(num*2)]
    b_matrix = [None for i in range(num*2)]
    for i in range(num):
        b_matrix[i*2] = [dst[i][0]]
        b_matrix[i*2 + 1] = [dst[i][1]]
        a_m = src[i]
        a_matrix[i*2] = [a_m[0],a_m[1],1,0,0,0,-a_m[0]*dst[i,0],-a_m[1] * dst[i,0]]
        a_matrix[i*2+1] = [0,0,0,a_m[0],a_m[1],1,-a_m[0] * dst[i,1],-a_m[1] * dst[i,1]]
    wrap = np.dot(np.mat(a_matrix).I,b_matrix)
    wrap = np.insert(wrap,num*2,1,0)
    wrap = wrap.reshape((3,3))
    return wrap

if __name__ == "__main__":
    src = np.array([[10, 457], [395, 291], [624, 291], [1000, 457]])
    dst = np.array([[46, 920], [46, 100], [600, 100], [600, 920.0]])
    wrap = getWrapMatrix(src,dst)
    print(wrap)

    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    wrap0 = getWrapMatrix(src,dst)
    print("wrap0 = ",wrap0)
    path = "../../image/photo1.jpg"
    img = cv2.imread(path)

    res_img1 = cv2.warpPerspective(img,wrap0,(327,488))
    cv2.imshow("Self_Wrap",res_img1)

    wrap1 = cv2.getPerspectiveTransform(src,dst)
    print("wrap1 = ",wrap1)
    res_img2 = cv2.warpPerspective(img,wrap1,(337,488))

    cv2.imshow("build_Wrap",res_img2)


    cv2.waitKey(0)


