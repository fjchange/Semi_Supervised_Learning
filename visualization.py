import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import array

G=nx.DiGraph()
with open('email-Eu-core.txt','r')as f:
    lines=f.readlines()
    num_node=lines.__sizeof__()
    M=dict()
    for line in lines:
        nodes=line.strip().split(' ')
        if M.has_key((nodes[0],nodes[1])):
            M[(nodes[0],nodes[1])]+=1
        else:
            M[(nodes[0],nodes[1])]=1
    for key in M:
        G.add_edge(key[0],key[1],weight=M.get(key))
    pos=nx.spring_layout(G)
    #degree=G.out_degree()
    #degrees=dict((x,y) for  x,y in degree)
    fig=plt.figure(figsize=(10,8),facecolor='white')
    nx.draw(G,pos,
            node_shape='o',edge_color='blue',width=0.5,with_labels=False,arrows=False,node_color='gray'
            )
    plt.show()