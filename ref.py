def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0] #样本数
    clusterAssment = mat(zeros((m,2))) #m*2的矩阵
    centroids = createCent(dataSet, k) #初始化k个中心
    clusterChanged = True
    while clusterChanged:      #当聚类不再变化
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k): #找到最近的质心
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i,0] != minIndex: clusterChanged = True
            # 第1列为所属质心，第2列为距离
            clusterAssment[i,:] = minIndex,minDist**2
        print centroids

        # 更改质心位置
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    # 这里第一列为类别，第二列为SSE
    clusterAssment = mat(zeros((m,2)))
    # 看成一个簇是的质心
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):    #计算只有一个簇是的误差
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2

    # 核心代码
    while (len(centList) < k):
        lowestSSE = inf
        # 对于每一个质心，尝试的进行划分
        for i in range(len(centList)):
            # 得到属于该质心的数据
            ptsInCurrCluster =\ dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            # 对该质心划分成两类
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算该簇划分后的SSE
            sseSplit = sum(splitClustAss[:,1])
            # 没有参与划分的簇的SSE
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
            # 寻找最小的SSE进行划分
            # 即对哪一个簇进行划分后SSE最小
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        # 较难理解的部分
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print 'the bestCentToSplit is: ',bestCentToSplit
        print 'the len of bestClustAss is: ', len(bestClustAss)
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment