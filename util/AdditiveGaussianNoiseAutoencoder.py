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
    
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input   
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights() #参数初始化
        self.weights = network_weights
        
        # model
        # 输入
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        # 隐含层
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale*tf.random_normal((n_input,)),
                                                     self.weights['w1']), self.weights['b1']))
        
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
        
        # cost
        self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
    
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.get_variable('w1',shape=[self.n_input, self.n_hidden],
                                            initializer=tf.contrib.layers.xavier_initializer())
            
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        return all_weights
    
    # 训练模型，训练过程优化，计算损失值    
    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={
                                      self.x:X,
                                      self.scale:self.training_scale})
    
    # 计算损失值    
    def calc_total_cost(self, X):
        return self.sess.run(self.cost,feed_dict={
                                      self.x:X,
                                      self.scale:self.training_scale})    
    # 向前计算隐含层的值
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X,
                                                       self.scale: self.training_scale
                                                       })    
    # 将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据    
    def generate(self, hidden=None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})
    
    # 整体运行一遍复原函数，包括提取高阶特征和通过高阶特征复原数据，即包括transform和generate。输入为原数据，输出为复原后的数据
    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict = {self.x: X,
                                                               self.scale: self.training_scale
                                                               })
        
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])    
        
        
        
        
        
