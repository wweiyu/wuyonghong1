# encoding=gbk
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt

class  LinearLeastSquareModel:

    def __init__(self,input,out_put):
        self.input_num = input
        self.output_num = out_put

    '用方程'
    def fit(self,data):
        xi = np.hstack(([data[:,i] for i in self.input_num]))
        yi = np.hstack(([data[:,j] for j in self.output_num]))
        num = data.shape[0]
        s_xy = np.sum(xi * yi)
        s_x= np.sum(xi)
        s_y = np.sum(yi)
        s_x2 = np.sum(xi * xi)
        e_x = s_x / num
        e_y = s_y / num
        a =(s_xy - e_y * s_x )/(s_x2 - e_x * s_x)
        b  = (s_x2 * e_y - s_xy * e_x)/(s_x2 - e_x * s_x)
        return [[b],[a]]

    # '用内置的函数求解'
    # def fit_build(self,data):
    #     xi = np.hstack(([data[:,i] for i in self.input_num]))
    #     yi = np.hstack(([data[:,j] for j in self.output_num]))
    #     xi = np.vstack((xi ** 0,xi)).T
    #     fit,_,r,s = sl.lstsq(xi,yi)
    #     return [[fit[0]],[fit[1]]]

    '计算误差 所有的误差平方和'
    def get_error(self,data,model):
        xi = np.vstack(([data[:,i] for i in self.input_num]))
        yi = np.vstack(([data[:,j] for j in self.output_num])).T
        n = data.shape[0]
        xi = np.vstack(([1] *n,xi)).T
        fit_b = np.dot(xi,model)
        errors = np.sum((fit_b - yi) ** 2,axis=1)
        return errors


def ransac(data,model,n,k,d,error):
    '''
    :param data:  数据
    :param model: 计算模型
    :param n: 每次选择的数量
    :param k: 重复次数
    :param d: 阈值 最少匹配的数量
    :param error: 每个值的平均误差
    :return: 最佳模型，所有内群点id
    '''
    best_fit = None
    iterations = 0
    best_error = np.inf
    num = data.shape[0]
    best_inner_ids = None
    while iterations < k:
        all_ids = np.arange(num)
        np.random.shuffle(all_ids)
        test_ids = all_ids[:n]
        other_ids = all_ids[n:]
        test_datas = data[test_ids,:]
        cur_fit = model.fit(test_datas)
        cur_all_erros = model.get_error(data[other_ids],cur_fit)
        '通过误差的内群数量'
        pass_ids = other_ids[cur_all_erros < error]
        if len(pass_ids) > d:
            cur_better_data = np.concatenate((test_datas,data[pass_ids,:]))
            cur_better_fit = model.fit(cur_better_data)
            cur_better_error = model.get_error(cur_better_data,cur_better_fit)
            cur_error = np.mean(cur_better_error)
            if best_error > cur_error:
                best_error = cur_error
                best_fit = cur_fit
                best_inner_ids = np.concatenate((test_ids,pass_ids))
        iterations += 1
    # if best_fit is None:
    #     raise ValueError("没有找到最佳模型")
    return best_fit,best_inner_ids

if __name__ == "__main__":
    m = 500
    exact_x = 20 * np.random.random(size=m)
    b, a = 5 * np.random.random(), 10 * np.random.normal()
    exact_y = exact_x * a + b
    print("exact fit :[{},{}]".format(b,a))
    '添加噪声'
    x_noise = exact_x + np.random.normal(size=m)
    y_noise = exact_y + np.random.normal(size=m)

    # '添加外群点'
    if True:
        out_lines = 100
        all_ids = np.arange(m)
        np.random.shuffle(all_ids)
        out_ids = all_ids[:out_lines]
        x_noise[out_ids] = 20 * np.random.random(size=out_lines)
        y_noise[out_ids] = 50 * np.random.normal(size=out_lines)

    '用最小二乘法求解'
    least_x = np.vstack((x_noise ** 0, x_noise))
    fit, resids, rank, s = sl.lstsq(least_x.T, y_noise)
    y_lease_fit = exact_x * fit[1] + fit[0]
    print("least_square fit = ", fit)

    '[xi,yi]'
    all_data = np.vstack((x_noise, y_noise)).T
    inputs = range(1)
    outputs = [i+1 for i  in range(1)]
    model = LinearLeastSquareModel(inputs,outputs)

    ransac_fit,ransac_inner_data = ransac(all_data,model,50,500,300,625)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x_noise, y_noise, 'k.', label='data')  # 散点图
    if not(ransac_fit is None):
        y_ransac_fit = exact_x * ransac_fit[1]+ ransac_fit[0]
        print("ransac fit = ",ransac_fit)
        ax.plot(x_noise[ransac_inner_data], y_noise[ransac_inner_data], 'cx', alpha=0.5, label='Ransac  data')
        ax.plot(exact_x, y_ransac_fit, 'orange', lw=2, label='Ransac fit y ={:.2f} + {:.2f}x'.format(ransac_fit[0][0], ransac_fit[1][0]))
    ax.plot(exact_x, exact_y, 'r', lw=2, label='True value y = {:.2f} + {:.2f}x'.format(b, a))
    ax.plot(exact_x, y_lease_fit, 'g', lw=2, label='Least square fit y ={:.2f} + {:.2f}x'.format(fit[1], fit[0]))
    ax.set_xlabel(r"$x$", fontsize=18)
    ax.set_ylabel(r"$y$", fontsize=18)
    ax.legend(loc=2)
    plt.show()
