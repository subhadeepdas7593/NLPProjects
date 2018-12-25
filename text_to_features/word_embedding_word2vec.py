"""
Author : Subhadeep Das
Title : To calculate average of the word's vector using pre-trained word2vec model
tools : Gensim
features used => word2vec
Description : The word's vector or average of that vector can be used as a feature for training of machine learning algorithms, in many applications
"""
from __future__ import division
from gensim.models import KeyedVectors 
import numpy as np

word_vectors = KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin.gz',binary=True)
print "\n loaded"
vector = word_vectors['reimbursement']
print vector

''' 
average of word vectors for a particular word 
This average can be used as a feature for training any algorithm

'''
avg = 0
add = 0
for n in vector:
	add += n 
avg = add / len(vector)
print "\n avg ==>> ",avg
print "\n numpy mean-->",np.mean(vector)
# print "np mean-->