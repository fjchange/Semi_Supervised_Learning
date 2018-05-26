import sklearn.tree as tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import copy

class Tree:
    def __init__(self):
        self.data=[]
        labels=[]
        with open('decision_tree_data_with_label.txt', 'r') as f:
            for line in f:
                linelist = line.split(' ')
                temp=copy.copy(linelist)
                labels.append(linelist[-1].strip())
                self.data.append(temp[0:-1])
        pca=PCA(n_components=100)
        self.data=pca.fit_transform(self.data)
        x = np.array(self.data)
        y= np.array(labels)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8)
        self.clf = tree.DecisionTreeClassifier(criterion='entropy')
        self.clf.fit(x_train, y_train)
        #print("clf:",str(self.clf))



        #precision, recall, thresholds = precision_recall_curve(y_train, self.clf.predict(x_train))
        #print precision, recall, thresholds
        #anwser = self.clf.predict_proba(x)[:, 1]
        #print(self.clf.predict(x_train))
        #print self.clf.score(x_train,y_train)
        #tree.export_graphviz(self.clf,out_file="tree.dot")

    def Rand_Index(self,gt_label,num_Node):
        A=B=C=D=0.
        t = self.clf.predict(self.data)
        for i in range(num_Node):
            for j in range(num_Node):
                if i!=j and t[i]==t[j] and gt_label[i][1]==gt_label[j][1]:
                    A+=1
                elif i!=j and t[i]==t[j] and gt_label[i][1]!=gt_label[j][1]:
                    B+=1
                elif i!=j and t[i]!=t[j]and gt_label[i][1]==gt_label[j][1]:
                    C+=1
                elif i!=j and t[i]!=t[j] and gt_label[i][1]!=gt_label[j][1]:
                    D+=1
        return (A+D)/(A+B+C+D)


