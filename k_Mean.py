#coding=utf-8
import numpy as np
import random
import math
class k_mean:
    def __init__(self,num_C,edges_dict,distance_matrix,max_iter=1000):
        self.K=num_C
        self.num_Node=edges_dict.__len__()
        # cal X 利用 node产生一个距离矩阵
        self.X=distance_matrix
        self.max_iter=max_iter


    def create_cen(self,k):
        cen_k=set()
        while len(cen_k)!=k:
            cen_k.add(random.randint(0,self.num_Node-1))
        return cen_k

    def train(self):
        centroids=self.create_cen(self.K)
        clusterChanged=True
        Y={}
        iter=0
        while clusterChanged and iter<self.max_iter:
            Y = {}
            clusterChanged = False
            for i in centroids:
                Y[i] = []
            for i in range(self.num_Node):
                #寻找最近的质心
                minDis=self.num_Node+1
                minIndex=-1
                for j in centroids:
                    if i!=j and self.X[i,j]<minDis:
                        minDis=self.X[i,j]
                        minIndex=j
                    elif i==j:
                        minIndex=i
                        break
                Y[minIndex].append(i)
            #print centroids

            #更新质心位置
            #通过公共子集的权重投票确定新的重心
            vote_prob=np.zeros((len(centroids),self.num_Node),dtype=np.float64)
            t=0
            for n in centroids:
                for node in Y[n]:
                    vote_prob[t]+=self.X[node]**2

                t+=1
            res=np.zeros(len(centroids),dtype=int)
            #res 新的质心点
            centroids_list=list(centroids)
            for i in range(len(centroids)):
                v_tmp=np.argmin(vote_prob[i,Y[centroids_list[i]]])
                res[i]=Y[centroids_list[i]][v_tmp]

            res=sorted(res)
            centroids_list=sorted(centroids_list)
            for i in range (len(centroids)):
                if res[i]!=centroids_list[i]:
                    clusterChanged=True
                    break
            if clusterChanged:
                centroids=set(res)
            iter+=1

        self.res_pre=np.ones(self.num_Node)*-1
        for i in range(len(centroids)):
            for node in Y[list(centroids)[i]]:
                self.res_pre[node]=i
        return centroids,Y

    def kMean(self,dataSet,k):
        centroids = self.create_cen(k)
        m=len(dataSet)
        Y = {}
        iter = 0
        clusterChanged=True
        res_before=set()
        clusterAssment=np.zeros((m,2))
        while clusterChanged:
            Y = {}
            clusterChanged = False
            for i in centroids:
                Y[i] = []
            for i in range(len(dataSet)):
                # 寻找最近的质心
                minDis = self.num_Node + 1
                minIndex = -1
                for j in centroids:
                    if dataSet[i] != j and self.X[dataSet[i], j] < minDis:
                        minDis = self.X[dataSet[i], j]
                        minIndex = j
                    elif dataSet[i] == j:
                        minIndex = dataSet[i]
                        break
                Y[minIndex].append(dataSet[i])
            #print centroids

            # 更新质心位置
            # 通过公共子集的权重投票确定新的重心
            vote_prob = np.zeros((len(centroids), m), dtype=np.float64)
            t = 0
            res = np.zeros(len(centroids), dtype=int)
            centroids_list = list(centroids)
            for n in centroids:
                temp=np.zeros(len(Y[n]),dtype=int)
                for node in Y[n]:
                    temp += self.X[node,Y[n]] ** 2
                res[t]=Y[n][np.argmin(temp)]

            res = sorted(res)
            centroids_list = sorted(centroids_list)
            for i in range(len(centroids)):
                if res[i] != centroids_list[i]:
                    clusterChanged = True
                    break
            if clusterChanged:
                centroids = set(res)
            iter += 1

        self.res_pre = np.ones(self.num_Node) * -1
        for i in range(len(centroids)):
            for node in Y[list(centroids)[i]]:
                self.res_pre[node] = i
        return centroids, Y

    def bikMean(self):
        #计算只有一个类的时候的效果
        clusterAssment=np.zeros((self.num_Node,2))
        centroid0=self.create_cen(1)
        centList=list(centroid0)
        for j in range(self.num_Node):
            clusterAssment[j,1]=self.X[j,centList[0]]**2
        Y={centList[0]:np.arange(0,self.num_Node)}

        iter=0
        while len(centList)<self.K and iter<self.max_iter:
            lowestSSE=1.0*self.num_Node*self.num_Node**2
            bestCentToSplit = None
            bestClutAss = None
            bestNewCents = None

            ignored_set=set()

            for i in range(len(centList)):
                if i not in ignored_set:
                    ptsIncurrCluster=Y[centList[i]]
                    centroidMat,splitClustAss=self.kMean(ptsIncurrCluster,2)
                    sseSplit=0
                    for cen in centroidMat:
                        t=self.X[splitClustAss[cen]]
                        sseSplit+=sum(t[:,cen]**2)
                    sseNotSplit=0

                    for j in range(len(centList)):
                        if i!=j:
                            t=self.X[Y[centList[j]]]
                            sseNotSplit+=sum(t[:,centList[j]]**2)

                    #print "sseSplit,and not sseNotSplit:",sseSplit,sseNotSplit
                    if len(centroidMat)==1:
                        ignored_set.add(i)
                    if len(centroidMat)!=1 and (sseSplit+sseNotSplit)<lowestSSE:
                        bestCentToSplit=i
                        bestNewCents=centroidMat
                        bestClutAss=splitClustAss
                        lowestSSE=sseSplit+sseNotSplit

            if bestCentToSplit!=None:
                Y.pop(centList[bestCentToSplit])
                Y=dict(bestClutAss.items()+Y.items())

                #print "+++++++> the bestCentToSplit is ",centList[bestCentToSplit]
                centList.remove(centList[bestCentToSplit])
                centList=centList+list(bestNewCents)
                #print 'the bestCentToSplit is ', bestClutAss
                #print 'the len of best ClustAss is',len(bestNewCents)
                print "------->",iter," the cent of cluster is ",centList
            iter+=1
        return centList,Y

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
