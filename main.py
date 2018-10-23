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

import time
start_time = time.clock()

import numpy as np
import itertools
import random

import sys

np.random.seed(42)
 

user_movie = np.load("user_movie.npy")
user_movie = user_movie[:10000000,:]

users_movies = dict() 
hash_dict = dict()

for i in user_movie:
    u_id = int(i[0])
    m_id = int(i[1])
    if u_id not in users_movies:
       users_movies[u_id] = {m_id} 
    else:
        users_movies[u_id].add(m_id) 
    if m_id not in hash_dict:
        hash_dict[m_id] = 0

all_movies = np.unique(user_movie[:,1])   
unique_users = len(np.unique(user_movie[:,0]))

sig_dict = dict()
sig_len = 100

for p in range(sig_len):
    print(p)
    index = np.random.permutation(max(all_movies)+1) 
        
    for u in users_movies:
        h = set()
        movies = users_movies[u]
        
        for m in movies:
            h.add(index[m])
 
        if u not in sig_dict:
            sig_dict[u] = [max(h)]
        else:
            sig_dict[u].append(max(h))
    
sig_list = list()                   
for u in range(unique_users):
    sig_list.append(sig_dict[u])
sig_mat = np.matrix(sig_list)
sig_mat = np.transpose(sig_mat)

bucket_dict = dict()
number_band = 10
for b in range(number_band):
    for u in range(unique_users):
        bucket_id = hash(sig_mat[10*b:10*b+10,u].tostring())
        if bucket_id not in bucket_dict:
            bucket_dict[bucket_id] = [u]
        else:
            bucket_dict[bucket_id].append(u)

bucket_dict = {k: v for k, v in bucket_dict.items() if len(v) > 1}

#partsize = 100
'''
for u_combo in itertools.combinations(sig_dict, 2):
    set_a = sig_dict[u_combo[0]]
    #set_a = set([set_a[i:i+partsize] for i in range(len(set_a))][:-partsize])
    set_b = sig_dict[u_combo[1]]
    #set_b = set([set_b[i:i+partsize] for i in range(len(set_b))][:-partsize])
    jdist = len(set_a & set_b) / len(set_a | set_b)
    combos = list()
    if jdist > 0.5:
        combos.append([u_combo, jdist])
'''
combos = list()        
for bucket_id, bucket in bucket_dict.items():
    combinations = itertools.combinations(bucket,2)
    for combination in combinations:
        vector_1 = user_movie[user_movie[:, 0] == combination[0]][:,1]
        vector_2 = user_movie[user_movie[:, 0] == combination[1]][:,1] 
        jdist = len(np.intersect1d(vector_1, vector_2)) / len(np.union1d(vector_1, vector_2))
        if jdist > 0.5:
            print([combination, jdist])
            combos.append([combination, jdist])

# https://www.learndatasci.com/tutorials/building-recommendation-engine-locality-sensitive-hashing-lsh-python/
# http://nbviewer.jupyter.org/github/mattilyra/LSH/blob/master/examples/Introduction.ipynb
# https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134

print(len(combos))
            
print("\n--- %s minutes ---" %((time.clock()- start_time)/60))
