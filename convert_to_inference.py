# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from train import Graph

def covert_inference(): 
    # Load graph
    output_path = 'logdir_inference'
    if not os.path.exists('logdir_inference'): 
    	os.mkdir('logdir_inference')	
    
    g = Graph(is_training=False)
    print("Graph loaded")
    # g = Graph(is_training=False)
    print(g)
    with tf.Session() as sess:
        tf.train.write_graph(g.graph, output_path, "graph.pbtxt", True)

                                          
if __name__ == '__main__':
    covert_inference()
    print("Done")
    
    