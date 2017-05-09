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
vocab_size = len(id_to_word)

def random_distribution():
    
    '''生成随机的概率列矩阵'''
    b = np.random.uniform(0.0, 1.0, size=[1, vocab_size])
    return b/np.sum(b, 1)[:,None]

