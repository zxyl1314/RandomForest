import random
import matplotlib.pyplot as plt
import numpy as np
import math
from typing import Union
import collections
from sklearn import datasets
from sklearn.model_selection import train_test_split

class DecisionNode(object):#决策树节点，用来存储子节点
    def __init__(self, f_idx = None, threshold = None, value = None):

        self.f_idx = f_idx
        self.threshold = threshold
        self.value = value
        self.L = None
        self.R = None
        pass

class MetaLearner(object):# 决策树
    def __init__(self,
                 min_samples : int=1,
                 min_gain   : float=0,
                 max_depth   : Union[int, None]=None,
                 max_leaves  : Union[int, None]=None):
        self.min_samples = min_samples #样本数少于 `min_smaples` 时，将当前节点作为叶子节点
        self.min_gain = 0.0 #增益小于 `min_gain` 时停止划分
        self.max_depth = max_depth #树递归的最大深度
        self.max_leaves = None #树的叶子节点的数目,超过这个数时，停止划分.
        self.head = None
        pass

    def fit(self, X : np.ndarray, y : np.ndarray) -> None:
        y = y. reshape(len(y), 1)
        Dataset = np.hstack((X,y))
        self.head = self.TreeNodeGenerate(Dataset)
        pass

    def predict(self, X):
        y=[]
        for i in X:
            node = self.head
            while not(node.L == None and node.R == None):
                if i[node.f_idx] < node.threshold:
                    node = node.L
                else:
                    node = node.R
            y.append(node.value)
        return np.array(y)

    def InForMationEntropy(self, Dataset: np.ndarray): #计算数据集信息熵
        if len(Dataset) == 0:
            return 0.0
        category_list = list(Dataset[:, -1])
        return np.var(category_list)

    def FindBestIdx(self, Dataset: np.ndarray):#寻找划分的最优属性下标，以及增益
        m = Dataset.shape[1] - 1
        k = int(math.log(m, 2)) + 1
        list_k = [i for i in range(Dataset.shape[1] - 1)]
        list_k = random.sample(list_k, k)
        best_idx = 0
        min_gain = 0.0
        best_gain = -math.inf #最大信息增益率,先设置为无穷小
        best_threshold = 0.0
        for i in list_k:
            present_threshold, present_gain = self.BestThreShold(Dataset, i) #寻找当前属性信息最优增益
            if present_gain > best_gain:
                best_gain = present_gain
                best_idx = i
                best_threshold = present_threshold

        return best_idx, best_threshold

    def BestThreShold(self, Dataset: np.ndarray, Idx): #寻找最优增益
        Entropy = self.InForMationEntropy(Dataset) #数据集信息熵
        attribute_list = list(set(Dataset[:, Idx]))
        best_gain = -math.inf #最大信息增益
        best_threshold = None #最优阈值
        for threshold in attribute_list:
            less_threshold = []
            more_threshold = []
            for (idx, d) in enumerate(Dataset[:, Idx]):
                if d < threshold:
                    less_threshold.append(idx)
                else:
                    more_threshold.append(idx)
            less_threshold = Dataset[less_threshold]
            more_threshold = Dataset[more_threshold]
            l_weight = len(less_threshold) / len(Dataset)
            m_weight = len(more_threshold) / len(Dataset)
            Entropysum = l_weight * self.InForMationEntropy(less_threshold) + \
                         m_weight * self.InForMationEntropy(more_threshold)
            gain = Entropy - Entropysum
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        return best_threshold, best_gain

    def FindMostClass(self,Class) -> float:#寻找类别最多的类
        return np.mean(Class)


    def TreeNodeGenerate(self, Dataset: np.ndarray, Depth = 1): #建树过程
        #print(Depth)
        if Depth < self.max_depth:
            if len(Dataset[:, -1]) <= self.min_samples or len(set(Dataset[:, -1])) == 1:
                Node = DecisionNode(value=list(set(Dataset[:, -1]))[0])
            elif Dataset.shape[1] == 1:
                Node = DecisionNode(value=self.FindMostClass(Dataset[:, -1]))
            else:
                best_idx,best_threshold = self.FindBestIdx(Dataset) #获取划分最优属性的下标以及阈值
                Node = DecisionNode(best_idx, best_threshold)
                L_data = Dataset[np.where(Dataset[:, best_idx] < best_threshold)]
                R_data = Dataset[np.where(Dataset[:, best_idx] >= best_threshold)]
                if len(L_data) == 0:
                    MostClass = self.FindMostClass(Dataset[:, -1])
                    Node.L = DecisionNode(value = MostClass)
                else:
                    Node.L = self.TreeNodeGenerate(L_data, Depth + 1)
                if len(R_data) == 0:
                    MostClass = self.FindMostClass(Dataset[:, -1])
                    Node.R = DecisionNode(value = MostClass)
                else:
                    Node.R = self.TreeNodeGenerate(R_data, Depth + 1)
        else:
            Node = DecisionNode(value=self.FindMostClass(Dataset[:, -1]))
        return Node

if __name__ == "__main__":
    iris = datasets.load_diabetes()
    X = iris.data
    y = iris.target.reshape((len(X), 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.44, random_state=42
    )

    print("------------init information-------------")
    print("the shape of X", X_train.shape)
    print("the shape of y", y_train.shape)
    print("the type of X", type(X))
    print("the type of y", type(y))

    print("---------------train log-----------------")
    forest_predict = []
    for i in range(50):
        #print("-------------------第",i, "次---------------------")
        Tree = MetaLearner(min_samples=5, max_depth=20)
        Tree.fit(X_train, y_train)
        predict = Tree.predict(X_test)
        # print("预测：", list(predict))
        # print("实际：", list(y_test[:, 0]))
        forest_predict.append(list(predict))
    print("---------------the result----------------")
    print(forest_predict)
    predict = []
    mse = []
    Len = len(y_test)
    temp = Len*[0]
    for i in range(50):
        predict_temp = []
        for j in range(len(y_test)):
            temp[j] += forest_predict[i][j]
            predict_temp.append(temp[j] / (i+1))
        sum = 0
        for j in range(len(predict_temp)):
            sum += (predict_temp[j] - y_test[j]) ** 2
        var_rate = sum / len(predict_temp)
        mse.append(var_rate[0])
    print("the rate of var is", var_rate[0])
    #-----------------------------------------------------
    # for i in range(len(y_test)):
    #     temp = 0
    #     for j in range(50):
    #         temp += forest_predict[j][i]
    #     temp = temp / 50
    #     predict.append(temp)
    #     sum = 0
    #     for j in range(len(predict)):
    #         sum += (predict[j] - y_test[j]) ** 2
    #     var_rate = sum / len(predict)
    #     mse.append(var_rate[0])
    #-------------------------------------------------------
    #predict = Tree.predict(X_test)
    # print("预测：", list(predict))
    # print("实际：", list(y_test[:, 0]))
    # sum = 0
    # for i in range(len(predict)):
    #     sum += (predict[i] - y_test[i]) ** 2
    # var_rate = sum / len(predict)
    #print("the rate of var is", var_rate[0])
    plt.plot(mse)
    plt.xlabel("the number of trees")
    plt.ylabel("MSE")

    plt.show()