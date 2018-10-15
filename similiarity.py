# -*- coding: utf-8 -*-

"""
Assignment 3: Similarity,
find pairs of users where jsim(u1, u2) > 0.5
- Jaccard similarity = intersect / union
- Using Locality sensitive hashing algorithm with MinHashing

- Input user_movie.npy data
- Also read random seed number from command line   
- Output to textfile; list of records in the form u1,u2
"""

import numpy as np
import itertools
import random

import sys

np.random.seed(42)
 

user_movie = np.load("user_movie.npy")


movie_users = dict() 

for i in user_movie:
    u_id = int(i[0])
    m_id = int(i[1])
    if m_id not in movie_users:
       movie_users[m_id] = {u_id} 
    else:
        movie_users[m_id].add(u_id)  

movies = list(movie_users)    
unique_users = u_id + 1
    
    
sig_dict = dict()

sig_len = 100
for p in range(sig_len):
    print(p)
    index = np.random.permutation(len(movies)) 
    for i in index:
        users_m = movie_users[ movies[i] ]
        for u in users_m:
            if u not in sig_dict:
                sig_dict[u] = {i}
            else:
                if len(sig_dict[u]) == p+1:
                    break
                else:
                    sig_dict[u].add(i) 
                
                 
#sig_mat = np.matrix(sig_list)
    
#partsize = 100

for u_combo in itertools.combinations(sig_dict, 2):
    set_a = sig_dict[u_combo[0]]
    #set_a = set([set_a[i:i+partsize] for i in range(len(set_a))][:-partsize])
    set_b = sig_dict[u_combo[1]]
    #set_b = set([set_b[i:i+partsize] for i in range(len(set_b))][:-partsize])
    jdist = len(set_a & set_b) / len(set_a | set_b)
    if jdist > 0.5:
        print(u_combo)
        
        
# https://www.learndatasci.com/tutorials/building-recommendation-engine-locality-sensitive-hashing-lsh-python/
# http://nbviewer.jupyter.org/github/mattilyra/LSH/blob/master/examples/Introduction.ipynb
# https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134
    



