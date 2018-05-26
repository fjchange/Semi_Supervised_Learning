import csv

def txt_dict(path):
    data=dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        num_node = lines.__sizeof__()
        for line in lines:
            nodes = line.strip().split(' ')
            if(data.has_key(int(nodes[0]))):
                data[int(nodes[0])].append(int(nodes[1]))
            else:
                data[int(nodes[0])]=[int(nodes[1])]
            if (data.has_key(int(nodes[1]))):
                data[int(nodes[1])].append(int(nodes[0]))
            else:
                data[int(nodes[1])]=[int(nodes[0])]
    return data

def csv_dict(path):
    data=dict()
    with open(path,'r')as f:
        reader=csv.reader(f)
        for row in reader:
            data[int(row[0])]=int(row[1])
    return data


def csv_list(path):
    data=list()
    with open(path,'r')as f:
        reader=csv.reader(f)
        for row in reader:
            for i in range(len(row)):
                row[i]=int(row[i])
            list.append(data,row)
    return data


