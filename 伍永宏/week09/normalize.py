import numpy as np
import matplotlib.pyplot as plt

def normalize_1(arr):
    '[0,1]'
    v_min = arr.min()
    v_max = arr.max()
    return (arr-v_min)/(v_max - v_min)

def normalize_2(arr):
    '[-1,1]'
    mean = np.mean(arr)
    return (arr-mean)/(arr.max() - arr.min())

def z_score(arr):
    'x∗=(x−μ)/σ'
    mean = np.mean(arr)
    sigma = np.std(arr)
    # sigma = np.sqrt(np.sum((arr-mean) * (arr-mean))/len(arr))
    # sigma = np.sum((arr-mean) * (arr-mean))/len(arr)
    return (arr-mean)/sigma

if __name__ == "__main__":
    arr =np.array([-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
         11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30])

    n1 = normalize_1(arr)
    n2 = normalize_2(arr)

    print(n1)
    print(n2)

    n3 = z_score(arr)
    counts = []
    for i in range(len(arr)):
        counts.append(np.sum(arr == arr[i]))

    plt.figure()
    plt.plot(arr,counts)
    plt.plot(n3,counts)
    plt.show()
