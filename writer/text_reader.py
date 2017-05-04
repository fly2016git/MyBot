#coding:utf-8
'''
Created on 2017年5月4日

@author: zpf_s
'''
import tensorflow as tf
import codecs
import os
import jieba
import re
import collections

def readfile(file_path):
    f = codecs.open(file_path, 'r', 'utf-8')
    text = f.read()
    text = re.sub(r'\s', '', text)
    seglist = list(jieba.cut(text, cut_all = False))
    return seglist

def _build_vocab(file_path):
    data = readfile(file_path)
    counter = collections.Counter(data)   # 统计词频
    count_pairs = sorted(counter.items(), key = lambda x : (-x[1], x[0]))
    
    words, _ = list(zip(*count_pairs)) # words和_分别代表zip后的两个部分
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict(zip(range(len(words)), words))
    dataids = []   # 训练集原文对应的词汇编号
    for w in data:
        dataids.append(word_to_id[w])
    
    return word_to_id, id_to_word, dataids


    
def dataproducer(batch_size, num_steps):
    word_to_id, id_to_word, data = _build_vocab('1.txt')
    datalen = len(data)
    batchlen = datalen//batch_size
    epcho_size = (batchlen - 1)//num_steps
    print(epcho_size)
    data = tf.reshape(data[0 : batchlen*batch_size], [batch_size, batchlen])
    print(data)
    i = tf.train.range_input_producer(epcho_size, shuffle=False).dequeue()
    print(i)
    x = tf.slice(data, [0, i*num_steps], [batch_size, num_steps])
    y = tf.slice(data, [0, i*num_steps+1], [batch_size, num_steps])
    x.set_shape([batch_size, num_steps])
    y.set_shape([batch_size, num_steps])
    return x, y, id_to_word

if __name__ == '__main__':
    x, y, id_to_word = dataproducer(100, 50)   
    print(x) 
    print(y) 
    
    
   
    
    
    
    
    
    
    
    
    