#coding:utf-8
'''
Created on 2017年4月12日

@author: 206
'''

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def xavier_init(fan_in, fan_out, constant = 1):
    '''
    一种参数初始化方法xavier initialization，可以根据输入、输出节点的数量调整最合适的分布，让权重初始化的不大不小，正好合适。
    fan_in:输入节点的数量
    fan_out：输出节点的数量
    '''
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval = low, maxval = high, dtype = tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    
    def __init__(self, n_input, n_hiden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input
        
        
        
        
        
