import csv
import random

def load_edge_list(path):
    c = open('decision_tree_data_with_label.csv', 'wb')
    writer = csv.writer(c, delimiter=',')
    with open(path, 'r') as f:
        lines = f.readlines()
        num_node = lines.__sizeof__()
        for line in lines:
            nodes = line.strip().split(' ')
            writer.writerow(nodes[1:])
    c.close()


load_edge_list('decision_tree_data_with_label.txt')

