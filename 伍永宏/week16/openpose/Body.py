from openpose.model.model import bodypose_model
from openpose.utils import *
from scipy.ndimage.filters import gaussian_filter
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Body():
    def __init__(self):
        self.model = bodypose_model()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        transfer(self.model,torch.load('./model_data/body_pose_model.pth'))
        self.model.eval()

    def __call__(self,image):
        image_shape = image.shape[:2]
        scale_search = [0.5]
        boxsize = 368
        stride = 8
        padValue = 128
        thre1 = 0.1
        thre2 = 0.05
        multiplier = [boxsize*s/image_shape[0] for s in scale_search]
        heatmap_avg = np.zeros(shape=(image_shape[0],image_shape[1],19))
        paf_avg = np.zeros(shape=(image_shape[0],image_shape[1],38))
        for m in multiplier:
            image_data = cv2.resize(image,None,fx=m,fy=m,interpolation=cv2.INTER_CUBIC)
            image_pad,pad = padRightDownCorner(image_data,stride,padValue)
            im = np.transpose(np.float32(np.expand_dims(image_pad,axis=0)),(0,3,1,2)) / 256.- 0.5
            im = np.ascontiguousarray(im)

            im_tensor = torch.from_numpy(im).float()
            with torch.no_grad():
                out_1,out_2 = self.model(im_tensor)
            out_1 = out_1.cpu().numpy()
            out_2 = out_2.cpu().numpy()

            heatmap = np.transpose(np.squeeze(out_2),(1,2,0))
            heatmap = cv2.resize(heatmap,(0,0),fx = stride,fy = stride,interpolation=cv2.INTER_CUBIC)
            heatmap = heatmap[:image_pad.shape[0] - pad[3],0:image_pad.shape[1] - pad[2],:]
            # heatmap = heatmap[:image_pad.shape[0] - pad[2],0:image_pad.shape[1] - pad[3],:]
            heatmap = cv2.resize(heatmap,(image_shape[1],image_shape[0]),interpolation=cv2.INTER_CUBIC)

            paf = np.transpose(np.squeeze(out_1),(1,2,0))
            paf = cv2.resize(paf,(0,0),fx=stride,fy= stride,interpolation=cv2.INTER_CUBIC)
            paf = paf[:image_pad.shape[0] - pad[3],0:image_pad.shape[1] - pad[2],:]
            # paf = paf[:image_pad.shape[0] - pad[2],0:image_pad.shape[1] - pad[3],:]
            paf = cv2.resize(paf,(image_shape[1],image_shape[0]),interpolation=cv2.INTER_CUBIC)

            heatmap_avg += heatmap_avg + heatmap / len(multiplier)
            paf_avg += paf/len(multiplier)

        all_keys = []
        key_counter = 0
        nPoints = 18
        for part in range(nPoints):
            cur_heatmap = heatmap_avg[:,:,part]
            one_heatmap = gaussian_filter(cur_heatmap,sigma=3)

            map_left = np.zeros(one_heatmap.shape)
            map_left[1:,:] = one_heatmap[:-1,:]

            map_right = np.zeros(one_heatmap.shape)
            map_right[:-1,:] = one_heatmap[1:,:]

            map_up = np.zeros(one_heatmap.shape)
            map_up[:,1:] = one_heatmap[:,:-1]

            map_down = np.zeros(one_heatmap.shape)
            map_down[:,:-1] = one_heatmap[:,1:]

            peaks_binary = np.logical_and.reduce((one_heatmap >= map_left,one_heatmap >= map_right,
                                                 one_heatmap >= map_up,one_heatmap >= map_down,one_heatmap >= thre1))

            peaks = list(zip(np.nonzero(peaks_binary)[1],np.nonzero(peaks_binary)[0]))
            peak_with_scores = [x + (cur_heatmap[x[1],x[0]],) for x in peaks]
            peak_id = range(key_counter,key_counter + len(peaks))
            peaks_with_score_and_id = [peak_with_scores[i] + (peak_id[i],) for i in range(len(peak_id))]
            all_keys.append(peaks_with_score_and_id)
            key_counter += len(peaks)
            # if part == 0:
            #     print('peak = ', peaks)
            #     print('peak_with_scores = ', peak_with_scores)
            #     print('peaks_with_score_and_id = ', peaks_with_score_and_id)
            #     break
        # image_copy = image.copy()
        # for part in range(nPoints):
        #     keys = all_keys[part]
        #     print(keys)
        #     for val in keys:
        #         print('val = ',val)
        #         cv2.circle(image_copy,(val[0],val[1]),2,(255,0,0))
        # cv2.imshow('keys',image_copy)
        # cv2.waitKey(0)
        '区分关键点'
        '同一根骨骼两端的两个关键点'
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                   [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                   [1, 16], [16, 18], [3, 17], [6, 18]]

        '代表与limbSeq对应的亲和场特征图索引'
        mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22], \
                  [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52], \
                  [55, 56], [37, 38], [45, 46]]
        connection_all = []
        special_k = []
        mid_num = 10
        for k in range(len(mapIdx)):
            cur_paf = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
            candA = all_keys[limbSeq[k][0]-1]
            candB = all_keys[limbSeq[k][1]-1]
            nA = len(candA)
            nB = len(candB)
            if nA != 0 and nB != 0:
                connection_candidate = []
                for i in range(nA):
                    for j in range(nB):
                        vec = np.subtract(candB[j][:2],candA[i][:2] )
                        norm = np.sqrt(vec[0]**2 + vec[1]**2)
                        norm = max(0.001,norm)
                        vec = np.divide(vec,norm)
                        '画一条直线过去，得到PAF上每个点的值'
                        interp_coord = list(zip(np.linspace(candA[i][0],candB[j][0],mid_num),
                                                np.linspace(candA[i][1],candB[j][1],mid_num)))
                        vec_1 = np.array([cur_paf[int(round(coord[1])),int(round(coord[0])),0] for coord in interp_coord])
                        vec_2 = np.array([cur_paf[int(round(coord[1])),int(round(coord[0])),1] for coord in interp_coord])
                        paf_scores = np.multiply(vec_1,vec[0]) + np.multiply(vec_2,vec[1])
                        ave_paf = sum(paf_scores)/len(paf_scores) + min(0.5 * image_shape[0] / norm - 1, 0)
                        criterion1 = len(np.nonzero(paf_scores > thre2)[0]) > 0.8 * len(paf_scores)
                        criterion2 = ave_paf > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, ave_paf, ave_paf + candA[i][2] + candB[j][2]])
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
                connection = np.zeros((0, 5))
                for c in range(len(connection_candidate)):
                    i, j, s = connection_candidate[c][0:3]
                    if i not in connection[:, 3] and j not in connection[:, 4]:
                        connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                        if len(connection) >= min(nA, nB):
                            break
                connection_all.append(connection)
            else:
                special_k.append(k)
                connection_all.append([])

        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_keys for item in sublist])
        for k in range(len(mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:, 0]
                partBs = connection_all[k][:, 1]
                indexA, indexB = np.array(limbSeq[k]) - 1

                for i in range(len(connection_all[k])):
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1

                    if found == 1:
                        j = subset_idx[0]
                        if subset[j][indexB] != partBs[i]:
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2:
                        j1, j2 = subset_idx
                        membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0:
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else:  # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)
        return candidate, subset


if __name__ == "__main__":
    body = Body()
    img_path = './imgs/ski.jpg'
    image = cv2.imread(img_path)
    candidate, subset =  body(image)
    canvas = draw_bodypose(image, candidate, subset)
    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.show()
