# coding=utf-8
import numpy as np

from data_process import data_convert
from softmax_regression import mini_batch_gradient_descent
from softmax_regression import softmax_regression
from softmax_regression import pytorch_softmax_regression
from sklearn.preprocessing import StandardScaler  # 导入数据标准化函数
import torch

def train(train_images, train_labels, k, iters = 5, alpha = 0.5):
    m, n = train_images.shape
    # data processing
    x, y = data_convert(train_images, train_labels, m, k) # x:[m,n], y:[1,m]
    print("x shape:", x.shape)
    print("y shape:", y.shape)

    # Initialize theta.  Use a matrix where each column corresponds to a class,
    # and each row is a classifier coefficient for that class.
    theta = np.random.rand(k, n) # [k,n]
    # do the softmax regression
    ## softmax
    #theta = softmax_regression(theta, x, y, iters, alpha)
    ## 小批次
    #batch_size=32;
    #theta=mini_batch_gradient_descent(theta,x,y,iters,alpha,batch_size,lam=0.001)


    theta=pytorch_softmax_regression(theta,x,y,iters=100,alpha=0.01)
    return theta

