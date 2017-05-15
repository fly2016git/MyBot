#coding:utf-8
'''
Created on 2017年5月9日

@author: 206
'''
import tensorflow as tf
from writer.text_reader import *
import numpy as np
import random

learning_rate = 1.0
num_steps = 35
hidden_size = 300
keep_prob = 1.0
lr_decay = 0.5
batch_size = 20
num_layers = 3
max_epoch = 14

x,y,id_to_word = dataproducer(batch_size, num_steps)
#vocab_size = len(id_to_word)
vocab_size = 10

def random_distribution():
    
    '''生成随机的概率列矩阵'''
    b = np.random.uniform(0.0, 1.0, size=[1, vocab_size]) #生成0到1之间的均匀分布
    return b/np.sum(b, 1)[:,None]

def sample_distribution(distribution):
    """
    Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    sample按照distribution的概率分布采样下标，这里的采样方式是针对离散的分布，相当于连续分布中求CDF(累计分布函数)。
    """
    r = random.uniform(0, 1) #随机生成一个0到1之间的数
    print(r)
    s = 0
    for i in range(len(distribution[0])): #累计求和,直至所有元素的和大于r值，返回此时的下标
        s += distribution[0][i]
        if s >= r:
            return i
    
    return len(distribution) - 1

def sample(prediction):
    """
    Turn a (column) prediction into 1-hot encoded samples.
            根据sample_distribution采样得的下标值，转换成1-hot的样本
    """
    d = sample_distribution(prediction)
    re = []
    re.append(d)
    return re

if __name__ == '__main__':
    distribution = random_distribution()
    print(distribution)
    #print(sample_distribution(distribution))
    print(sample(distribution))