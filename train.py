#coding=utf-8

import preprocessing as pre
from LPA_with_dict import LPA_dict as lpa
import numpy as np
import decesion_Tree
import random
import k_Mean
labeled_data_path='labeled_node.txt'
gt_labeled_data_path='email-Eu-core-department-labels.txt'
edges_path='email-Eu-core.txt'
bidding_list_path='bidding-list.txt'
distance_matrix_path="distance_matrix.txt"

#lpa_D=lpa.LPA_dict(labeled_data=pre.csv_dict())

#labeled_data=np.loadtxt(labeled_data_path,dtype=np.uint16)

labeled_percent=0.05
gt_labeled_data=np.loadtxt(gt_labeled_data_path,dtype=np.uint16)
tmp_label_data=[]
edges=pre.txt_dict(edges_path)
X=np.loadtxt(distance_matrix_path,dtype=int)

num_Node = edges.__len__()
# cal X 利用 node产生一个距离矩阵
'''
X = np.ones([num_Node, num_Node], dtype=np.int32)*num_Node
for i in range(num_Node):
    for index in edges[i]:
        if i != index:
            X[i][index] = 1
        else:
            X[i][index] = 0
        for j in range(i):
            if X[j][i] + X[i][index] < X[j][index]:
                X[j][index] = X[j][i] + X[i][index]

f=open ("decision_tree_data_with_label.txt",'w')
for i in range(num_Node):
    f.write(str(i)+' ')
    for j in range(num_Node):
        f.write(str(X[i][j])+' ')
    f.write(str(gt_labeled_data[i][1])+'\n')
f.close()
'''
for i in range(5):
    for i in gt_labeled_data:
        if random.random()<labeled_percent:
            tmp_label_data.append(i)
    labeled_data=np.array(tmp_label_data)
    lpa_D=lpa(labeled_data=labeled_data,
              edges_dict=edges,
              num_C=42,
              distance_matrix=X,
              determine_mode=-1,
              max_iter=500,
              bidding_list=np.loadtxt(bidding_list_path,dtype=np.uint16)
              )
    lpa_D.train()
    #print(result)
    print(lpa_D.Rand_Index(gt_labeled_data))

'''
k_mean=k_Mean.k_mean(num_C=42,edges_dict=edges,distance_matrix=X)
#_,res=k_mean.bikMean()
k_mean.train()
#print(res)
print(k_mean.Rand_Index(gt_labeled_data))

dt=decesion_Tree.Tree()
print(dt.Rand_Index(gt_labeled_data,len(edges.keys())))
'''