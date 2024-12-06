# 学习手动实现神经网络

import numpy as np
import scipy.special

class NeuralNet:
    def __init__(self,input_nodes,output_nodes,learn_rate = 0.1):
        self.len_layer = 2
        self.out_nodes = output_nodes
        self.layers = [input_nodes]
        self.weights = []
        self.bias = []
        self.lr = learn_rate

    def add_hide_layer(self,hide_nodes):
        self.layers.append(hide_nodes)
        '添加权重与偏置'
        per_notes = self.layers[-2]
        self.weights.append(np.random.rand(per_notes,hide_nodes) - 0.5)
        self.bias.append(np.random.rand())
        # self.bias.append(0)
        self.len_layer += 1

    def _check_layers(self):
        if len(self.layers) == self.len_layer:
            return
        self.layers.append(self.out_nodes)
        per_notes = self.layers[-2]
        self.weights.append(np.random.rand(per_notes, self.out_nodes) - 0.5)
        self.bias.append(np.random.rand())
        # self.bias.append(0)


    def train(self,data,label):
        self._check_layers()
        'forward'
        cur_data = data[np.newaxis,:]
        input_nodes = []
        output_nodes  = []
        for i in range(self.len_layer-1):
            cur_input = np.dot(cur_data,self.weights[i]) + self.bias[i]
            cur_output = scipy.special.expit(cur_input)
            input_nodes.append(cur_input)
            output_nodes.append(cur_output)
            cur_data = cur_output
            # print(f" forward - input : {cur_input.shape},output : {cur_output.shape}")

        # print("out_put = ",output_nodes[-1])
        'loss'
        loss = self._getloss(output_nodes[-1],label)
        print('loss = ',loss)

        'backward'
        lst_des = []
        for i in range(self.len_layer-2,-1,-1):
            out_d = output_nodes[i]
            pre_out_d = output_nodes[i-1]
            # 最后一层
            var = out_d * (1 - out_d)
            if i == self.len_layer-2:
                var = var * (out_d - label)
            else:
                var = var * lst_des
                if i == 0:  # 第0层
                    pre_out_d = data[np.newaxis, :]
            des = np.dot(pre_out_d.T, var)
            if i > 0 :
                lst_des = np.dot(self.weights[i], var.T).T
            # print('des :',des.shape)
            self.weights[i] = self.weights[i] - self.lr * des
            # bias_var = np.dot(var, var.T)
            # print('bias =',bias_var)
            # self.bias[i] = self.bias[i] - self.lr * bias_var

    def query(self,data):
        cur_data = data[np.newaxis,:]
        input_nodes = []
        output_nodes = []
        for i in range(self.len_layer-1):
            cur_input = np.dot(cur_data,self.weights[i]) + self.bias[i]
            cur_output = scipy.special.expit(cur_input)
            input_nodes.append(cur_input)
            output_nodes.append(cur_output)
            cur_data = cur_output
        return output_nodes[-1]

    def _getloss(self,output,label):
        loss =  np.mean(np.square(output - label))
        return loss

if __name__ == "__main__":
    output = 10
    net = NeuralNet(784,output)
    net.add_hide_layer(512)
    net.add_hide_layer(100)
    epoch = 50
    for i in range(epoch):
        with open("../../../dataset/mnist_train.csv") as ff:
            res = ff.readlines()
            for record in res:
                data = record.split(',')
                input_data = (np.asfarray(data[1:]))/255.0 * 0.99 + 0.11
                label = np.zeros(output) + 0.11
                label[int(data[0])] = 0.99
                net.train(input_data, label)

    with open("../../../dataset/mnist_test.csv") as tt:
        res = tt.readlines()
        total = len(res)
        accuracy = 0
        for record in res:
            data = record.split(',')
            in_data = np.asfarray(data[1:])/255.0 * 0.99 + 0.11
            label = int(data[0])
            output = net.query(in_data)
            if label == np.argmax(output):
                accuracy += 1

        print("accuracy : ",accuracy/total)
