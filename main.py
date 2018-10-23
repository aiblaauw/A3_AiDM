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
from scipy.sparse import csr_matrix, csc_matrix

## randomseed = int(sys.argv[1])
## filepath = sys.argv[2]

np.random.seed(42)

start_time = time.clock()
 
user_movie = np.load("user_movie.npy")

num_movies = len(np.unique(user_movie[:,1]))
unique_users = len(np.unique(user_movie[:,0]))

sig_len = 100

sparse_matrix = csr_matrix((np.ones(len(user_movie)), (user_movie[:,1], user_movie[:,0])))
sig_mat = np.array(range(unique_users))
for p in range(sig_len):
    print(p)
    index = np.random.permutation(num_movies) 
    sparse_matrixcsc = sparse_matrix[index,:].tocsc()   
    first_nonzero_row = sparse_matrixcsc.indices[sparse_matrixcsc.indptr[:-1]]
    sig_mat = np.vstack((sig_mat, first_nonzero_row))
    
    
print("\n--- %s minutes ---" %((time.clock()- start_time)/60))


number_band = 2
potential_combos = list()
for b in range(number_band):
    bucket_dict = dict()
    for u in range(unique_users):
        bucket_id = hash(sig_mat[np.int(sig_len/number_band)*b:np.int(sig_len/number_band)*b+np.int(sig_len/number_band),u].tostring())
        if bucket_id not in bucket_dict:
            bucket_dict[bucket_id] = [u]
        else:
            bucket_dict[bucket_id].append(u)
    bucket_dict = {k: v for k, v in bucket_dict.items() if len(v) > 1}
    for bucket_id, bucket in bucket_dict.items():
        combinations = itertools.combinations(bucket,2)
        for combination in combinations:
            potential_combos.append(combination)

combocount = 0
for combination in potential_combos:
    vector_1 = user_movie[user_movie[:, 0] == combination[0]][:,1]
    vector_2 = user_movie[user_movie[:, 0] == combination[1]][:,1] 
    jdist = len(np.intersect1d(vector_1, vector_2)) / len(np.union1d(vector_1, vector_2))
    if jdist > 0.5:
        combocount += 1
        print("\n--- %s minutes ---" %((time.clock()- start_time)/60))      
        print([combocount, combination, jdist])
        ## output = open("results.txt", "a")
        ## output.write(str(combination[0]) + "," + (combination[1]))
        ## output.close()

        
# https://www.learndatasci.com/tutorials/building-recommendation-engine-locality-sensitive-hashing-lsh-python/
# http://nbviewer.jupyter.org/github/mattilyra/LSH/blob/master/examples/Introduction.ipynb
# https://towardsdatascience.com/understanding-locality-sensitive-hashing-49f6d1f6134
            
print("\n--- %s minutes ---" %((time.clock()- start_time)/60))
