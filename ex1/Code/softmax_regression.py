# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn  # 导入PyTorch的神经网络模块
from sklearn.preprocessing import StandardScaler  # 导入数据标准化函数
# 定义神经网络类
class net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(net, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        

    def forward(self, x):
        out = self.layer(x)
        return out  # 返回最终输出


def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    #损失值f
    f=list()
    lam=0.001
    data=x.T
    accuracy_list=list()
    for i in range (iters):
        x=np.dot(theta,data) # x=权重*样本数据
        exp_x=np.exp(x) # e^x

        exp_x_sum=exp_x.sum(axis=0)
        ## 交叉熵
        y_hat=(exp_x/exp_x_sum) #y^

        loss=0.0 
        log_y_hat=np.log(y_hat)
        ## 计算损失函数
        for j in range(y.shape[1]):
            loss+=np.dot(log_y_hat[:,j],y_hat[:,j]) ## loss =xigemalogy *y

        # 平均值
        train_loss=-(1.0/y.shape[1])*loss
        print("train_loss:",train_loss)
        f.append(train_loss)

        ## 计算梯度
        batch_size=y.shape[1]
        g=-(1.0/batch_size)*np.dot((y-y_hat),data.T)+lam*theta
        theta=theta-alpha*g
        
    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.arange(iters), f)
    plt.show()


    return theta

def mini_batch_gradient_descent(theta, x, y, iters, alpha, batch_size, lam=0.001):
   #  的形状是 (60000, 784)，y 的形状是 (10, 60000)

    # 定义超参数
    batch_size = 64  # 小批次大小
    num_samples = x.shape[0]  # 样本总数
    num_batches = num_samples // batch_size  # 总批次数
    f=list()
    # 随机初始化模型参数 theta，假设 theta 的形状是 (10, 7lam84)
    theta = np.random.rand(10, 784)

    # 定义学习率和迭代次数等超参数
    learning_rate = 0.01
    num_epochs = 100

    # 迭代训练
    for epoch in range(num_epochs):
        # 随机打乱数据集的顺序
        # permutation = np.random.permutation(num_samples)
        # x_shuffled = x[permutation]
        # y_shuffled = y[:, permutation]
 
        for batch in range(num_batches):
            # 获取当前小批次数据
            start = batch * batch_size
            end = (batch + 1) * batch_size
            x_batch = x[start:end]
            y_batch = y[:, start:end]

            # 执行前向传播，计算损失，计算梯度
            logits = np.dot(theta, x_batch.T)
            exp_x = np.exp(logits)
            exp_x_sum = np.sum(exp_x, axis=0)
            y_hat = exp_x / exp_x_sum
            loss = -np.sum(y_batch * np.log(y_hat)) / batch_size
            gradient = -(1.0 / batch_size) * np.dot((y_batch - y_hat), x_batch) +lam*theta # 计算梯度
           
            # 更新模型参数
            theta -= learning_rate * gradient
        f.append(loss)
        # 打印损失或其他训练过程中的指标
        print(f"Epoch {epoch+1}, Loss: {loss}")
    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.arange(num_epochs), f)
    plt.show()

    return  theta 

