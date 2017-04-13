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


def standard_scale(X_train, X_test):
    '''
    对训练数据和测试数据进行标准化处理
    '''
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    '''
    随机获取数据块
    '''
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

mnist = input_data.read_data_sets('MNIST_data', one_hot=True) 

def run(X_train=mnist.train.images, X_test=mnist.test.images):
    # 标准化变换
    X_train, X_test = standard_scale(X_train, X_test)
    # 定义参数：总训练样本数、最大训练轮数、batch_size , 并设置每隔一轮显示一次cost
    n_samples = int(mnist.train.num_examples)
    training_epochs = 20
    batch_size = 128
    display_step = 1
    
    # 创建自编码器实例
    autoencoder = AdditiveGaussianNoiseAutoencoder(
        n_input=784,
        n_hidden = 200,
        transfer_function=tf.nn.softplus,
        optimizer= tf.train.AdamOptimizer(learning_rate=0.001),
        scale=0.01                                 )
    
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(n_samples/batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples*batch_size
            
        if epoch%display_step == 0:
            print('Epoch:', '%04d' % (epoch+1), 'cost=', "{:.9f}".format(avg_cost))
            
    print("total cost: " + str(autoencoder.calc_total_cost(X_test)))
            

class AdditiveGaussianNoiseAutoencoder(object):
    '''
    去噪自编码器
    '''
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
        self.n_input = n_input   
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale   # 噪声系数
        network_weights = self._initialize_weights() # 参数初始化
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
    # 获取隐含层的权重w1    
    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    # 获取隐含层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])    
        
       
           
        
