import  cv2
import numpy as np
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster
import matplotlib.pylab as plt

def is_same_class(flags,idx1,idx2):
    if flags.get(idx1) and flags.get(idx2):
        return flags[idx1] == flags[idx2]
    return False

def hiercluster_detail(data):
    flags = dict()
    length = len(data)
    '类与类之间用最短距离法'
    back = []
    count = 0
    cur_index = length
    while count < data.shape[0]:
        min_dis = -1
        idx1,idx2 = -1,-1
        for i in range(length):
            for j in range(i+1,length):
                '同类不做处理'
                if is_same_class(flags,i,j):
                    continue
                dis = np.sqrt((data[i][0] - data[j][0]) ** 2 + (data[i][1] - data[j][1]) ** 2)
                if idx1 < 0 or min_dis > dis:
                    min_dis = dis
                    idx1,idx2 = i,j
        if flags.get(idx1) and flags.get(idx2):
            tag1 = flags[idx1]
            tag2 = flags[idx2]
            flags[cur_index] = flags[tag1] + flags[tag2]
            flags[idx1] = cur_index
            flags[idx2] = cur_index
            for var in flags[tag1]:
                '变更类'
                flags[var] = cur_index
            for var in flags[tag2]:
                '变更类'
                flags[var] = cur_index
            back.append([tag1,tag2,min_dis, len(flags[cur_index])])
            del flags[tag1]
            del flags[tag2]
        elif flags.get(idx1):
            old_tag = flags[idx1]
            num = 0
            for var in flags[old_tag]:
                '变更类'
                num += 1
                flags[var] = cur_index
            back.append([idx2,old_tag,min_dis, num+1])
            flags[cur_index] = [var for var in flags[old_tag]]
            flags[cur_index].append(idx2)
            flags[idx2] = cur_index
            del flags[old_tag]
        elif flags.get(idx2):
            old_tag = flags[idx2]
            num = 0
            for var in flags[old_tag]:
                '变更类'
                num += 1
                flags[var] = cur_index
            back.append([idx1,old_tag,min_dis, num+1])
            flags[cur_index] = [var for var in flags[old_tag]]
            flags[cur_index].append(idx1)
            flags[idx1]= cur_index
            del flags[old_tag]
        else:
            back.append([idx1, idx2, min_dis, 2])
            flags[cur_index] = [idx1,idx2]
            flags[idx1] = cur_index
            flags[idx2] = cur_index
        cur_index += 1
        count = back[-1][3]
    return back


if __name__ == "__main__":
    # x = np.array([[1,2],[3,2],[4,4],[1,2],[1,3]])
    x = np.array(
        [[0.0888, 0.5885],
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
         ])
    back = hiercluster_detail(x)
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    dendrogram(back)

    zz = linkage(x, method='single')
    ff = fcluster(zz, 2, 'distance')
    plt.subplot(122)
    dendrogram(zz)

    plt.show()
