#coding=utf-8

import numpy as np
import math

def get_weight(a,b,sigma,dist_func):
    return math.exp(-1*(sum(np.dot(dist_func(a,b),dist_func(a,b))))/sigma**2)

class LPA:
    #需要注意过多的中间矩阵容易导致存储空间不足，尽量压缩需要的存储空间，Linux分配单个进程空间有4G
    def __init__(self,labeled_data,labeled_list,unlabeled_data,label_list,dis_func):
        self.X_l=labeled_data
        #one hot 矩阵
        self.Y_l=np.eye(label_list.size)[labeled_list]
        #对于unlabeled的data来说，分配的类不影响
        self.X_u=unlabeled_data
        self.C=label_list
        self.l=self.X_l.shape[0]
        self.u=self.X_u.shape[0]
        #C是所有类的个数
        self.num_C=label_list.size
        #合并label_data和unlabel_data
        self.X=np.append(labeled_data,unlabeled_data,axis=0)
        #以0为Y_0，默认初始方法
        Y_u=np.zeros((unlabeled_data.shape[0],self.num_C))
        self.Y=self.Y_l.append(Y_u)

        #转移概率矩阵
        self.T=np.zeros((self.l+self.u),dtype=float)
        #因为并不是所有的数据计算距离方式都一样，需要外部引入
        self.dis_func=dis_func

    def cal_T(self,sigma):
        for i in range(self.l+self.u):
            for j in range(self.l+self.u):
                self.T[i,j]=get_weight(self.X[i],self.X[j],sigma,self.dis_func)
            self.T[i]=self.T[i]/sum(self.T[i])

    def train(self):
        iter=0
        while(1):
#1006x1006 1006x42
            res=np.matmul(self.T,self.Y)
            if np.equal(self.Y,res):
                #相等意味着收敛
                break
            else :
                if iter%10==0:
                    new=np.copy(res)
                    new[:self.l]=np.copy(self.Y_l)
                    changed=sum(res-new)
                    self.Y=new
                    print "-----> iteration %d,changed:%f"%(iter,changed)
                else:
                    self.Y=np.copy(res)
                    self.Y[:self.l]=np.copy(self.Y_l)

            iter+=1
        return self.label_determine()

    def label_determine(self):
        res=np.zeros(self.l+self.u)
        res=np.argmax(self.Y,axis=1)
        self.res_pre=res
        return self.res_pre

    def cal_accur(self,gt_label,pred_label):
        ac=0
        for i in range(self.l+self.u):
            if 
