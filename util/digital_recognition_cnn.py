#coding:utf-8
'''
Created on 2017年4月10日

@author: 206
'''

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', '../data', 'Directory for storing data')

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

#初始化生成随机变量的权重，避免神经元输出恒为零
def weight_variable(shape):
    #以正态分布生成随机值
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 初始化生成随机的偏置项(常量)，避免神经元输出恒为0
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)#所有值均为0.1
    return tf.Variable(initial)



