#coding=utf-8
import math
import numpy as np

'''
@ author: kiwi_fung
@ time：2018/05/08
@ last update :2018/05/10
'''

class LPA_dict:
    '''
    输入数据要求：
    labeled_data：已经标记的节点号 类号
    num_C:社团的个数
    data:所有的边数据
    determine_node：为解决分类依赖于已经标记节点的比例的模式
    CMP_pro：用于调节权重的系数，超参数，用np.array输入
    bidding_list:为方便使用，需输入的为dict/array/matrix
    '''

    #输入已经分类的数据、数据的类数、未分类的数据,格式array,需要转换为dict
    def __init__(self,labeled_data,num_C,edges_dict,determine_mode,distance_matrix,CMP_pro=None,bidding_list=None,max_iter=1000,min_limit=0.0001):
        self.l = len(labeled_data)
        self.labeled_data=labeled_data
        #类的个数
        self.num_C=num_C

        self.num_Node=edges_dict.__len__()
        self.X=distance_matrix
        self.Y=np.zeros([self.num_Node,self.num_C],dtype=np.float64)
        self.Y=self.set_Y_l(self.Y)

        self.u=self.num_Node-self.l
        #用于距离权重的转化超参数
        self.sigma=self.cal_d_0(labeled_data[:,0])
        self.determine_mode=determine_mode
        self.CMP_pro=CMP_pro
        self.bidding_list=bidding_list
        self.min_limit=min_limit
        self.max_iter=max_iter

    def set_Y_l(self,Y):
        for node in self.labeled_data:
            Y[node[0]]=np.zeros(self.num_C,dtype=np.float64)
            Y[node[0]][node[1]]=1.
        return    Y
    #d_0 来源于通过对已知最近的类间距离计算（最小生成树
    def cal_d_0(self,labeled):
        min_dis=self.num_Node
        for i in labeled:
            for j in labeled:
                if i!=j and self.X[i][j]<min_dis:
                    min_dis=self.X[i][j]
                if min_dis==1 :break
            if min_dis==1 :break
        return min_dis/3.

    #计算T矩阵
    def cal_T(self):
        self.T=np.exp(-1*np.multiply(self.X,self.X)/self.sigma**2)
        if self.determine_mode==0:
            self.T=np.multiply(self.T,np.repeat(self.CMP_pro,(self.num_Node,1)))
        t_sum=np.transpose([np.sum(self.T,axis=0)])
        t_sum=np.repeat(t_sum,self.num_Node,axis=1)
        self.T=np.true_divide(self.T,t_sum)

    def train(self):
        iter=0
        self.cal_T()
        while iter<self.max_iter:
            res=np.matmul(self.T,self.Y)
            self.set_Y_l(res)
            changed = np.sum(res - self.Y)
            if (changed<=self.min_limit):
                print("已然收敛！")
                break
            elif(iter%100==0):
                self.Y=res
                #print("----->iteration %d,changed: %f")%(iter,changed)
            else:
                self.Y=res
            iter+=1
        return self.label_determine()

    def label_determine(self):
        res =np.ones((self.num_Node,1))*-1
        if self.determine_mode==1:

            # which means use label_bidding
            for i in range(self.num_C-1,-1,-1):
                while self.bidding_list[i]>0:
                    node_max=np.argmax(self.Y[:,i],axis=0)
                    res[node_max]=i
                    self.bidding_list[i]-=1
                    self.Y[node_max]=np.ones(self.num_C)*-1

        else:
            res=np.argmax(self.Y,axis=1)
        self.res_pre=res
        return self.res_pre

    def cal_accur(self,gt_label):
        #对预测对的数据进行计数
        ac=0.
        for i in range(self.num_Node):
            if gt_label[i][1]==self.res_pre[i]:
                ac+=1
        return ac/self.num_Node

    def Rand_Index(self,gt_label):
        A=B=C=D=0.
        for i in range(self.num_Node):
            for j in range(self.num_Node):
                if i!=j and self.res_pre[i]==self.res_pre[j] and gt_label[i][1]==gt_label[j][1]:
                    A+=1
                elif i!=j and self.res_pre[i]==self.res_pre[j] and gt_label[i][1]!=gt_label[j][1]:
                    B+=1
                elif i!=j and self.res_pre[i]!=self.res_pre[j] and gt_label[i][1]==gt_label[j][1]:
                    C+=1
                elif i!=j and self.res_pre[i]!=self.res_pre[j] and gt_label[i][1]!=gt_label[j][1]:
                    D+=1
        return (A+D)/(A+B+C+D)

