#coding:utf-8
'''
Created on 2017年5月9日

@author: 206
'''
import tensorflow as tf
from writer.text_reader import *
import numpy as np
import random


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
    返回下标 
    """
    d = sample_distribution(prediction)
    re = []
    re.append(d)
    return re


learning_rate = 1.0
num_steps = 35
hidden_size = 300
keep_prob = 1.0
lr_decay = 0.5
batch_size = 20
num_layers = 3
max_epoch = 14

x,y,id_to_word = dataproducer(batch_size, num_steps)
vocab_size = len(id_to_word)
#vocab_size = 10

size = hidden_size
#lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias = 0.5) # froget_bias是gorget gate的值
#lstm_cell = tf.nn.rnn_cell.DropoutWrapper()
lstm_cell = tf.contrib.rnn.BasicLSTMCell(size, forget_bias = 0.5)
lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = keep_prob)
# 使用堆叠函数将多层lstm_cell堆叠到cell中， num_layers为堆叠层次
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell], num_layers)
# 设置LSTM单元的初始化状态为0
initial_state = cell.zero_state(batch_size, tf.float32)
state = initial_state
# 生成词向量
embedding = tf.get_variable("embedding", [vocab_size, size])

input_data = x
targets = y

test_input = tf.placeholder(tf.int32, shape=[1])
test_initial_state = cell.zero_state(1, tf.float32)
# 查询单词对应的向量表达，获取inputs
inputs = tf.nn.embedding_lookup(embedding, input_data)
test_inputs = tf.nn.embedding(embedding, test_input)

outputs = []
initializer = tf.random_uniform_initializer(-0.1, 0.1)
with tf.variable_scope("Model", reuse = None, initializer = initializer):
    with tf.variable_scope("r", reuse= None, initializer = initializer):
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        sotfmax_b = tf.get_variable("softmax_w", [vocab_size])

if __name__ == '__main__':
    distribution = random_distribution()
    print(distribution)
    #print(sample_distribution(distribution))
    print(sample(distribution))