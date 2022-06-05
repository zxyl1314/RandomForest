import numpy as np
import math
import graphviz
from typing import Union
import collections
from sklearn import datasets
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd
from texttable import Texttable

class DecisionNode(object):#决策树节点，用来存储子节点

    """
        参数解释：
            - f_idx : 属性的下标，可以使用该下标得到具体的属性
            - threshold : 下标 `f_idx` 对应属性的阈值
            - value : 如果该节点是叶子节点，对应的是被划分到这个节点的数据的类别（分类决策树），可以用整数数值进行表达
            - L : 左子树
            - R : 右子树
    """

    def __init__(self, f_idx = None, threshold = None, value = None):

        self.f_idx = f_idx
        self.threshold = threshold
        self.value = value
        self.L = None
        self.R = None
        pass

class SimpleDecisionTree(object):# 决策树
    def __init__(self,
                 min_smaples : int=1,
                 min_gain    : float=0,
                 max_depth   : Union[int, None]=None,
                 max_leaves  : Union[int, None]=None):
        min_smaples = 1 #样本数少于 `min_smaples` 时，将当前节点作为叶子节点
        min_gain = 0.0 #增益小于 `min_gain` 时停止划分
        max_depth = None #树递归的最大深度
        max_leaves = None #树的叶子节点的数目,超过这个数时，停止划分.
        self.head = None
        pass

    def fit(self, X : np.ndarray, y : np.ndarray, way_i) -> None:
        """
        参数解释
        ------
        X : 训练数据，数据的维度为 (n, m), n 是样本个数，m 是属性个数
        y : X 当中的 n 个样本的 类别，y中的数值类型为整型
        """
        #y = y. reshape(len(y), 1)
        Dataset = np.hstack((X,y))
        self.head = self.TreeNodeGenerate(Dataset, way_i)
        pass

    def predict(self, X):
        """
      参数解释
      ------
       X : 测试数据，数据的维度为 (n, m), n 是样本个数，m 是属性个数

      Return
      ------
      返回的数据类型是 np.ndarray，ndarray 里面的数据类型是 np.int64 ，数据维度为 (n, )
      """
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
        Entropy = 0.0
        category_list = list(Dataset[:, -1])
        for category in set(category_list):
            Pk = category_list.count(category) / len(Dataset)
            Entropy -= Pk * math.log(Pk, 2) #信息熵公式
        return Entropy

    def InformationGainRate(self,Dataset: np.ndarray, F_idx, way_i): #计算某一属性信息增益率
        Entropy = self.InForMationEntropy(Dataset) #信息熵
        EntropySum = 0.0 #信息增益
        Iv_x = 0.0
        Attribute_list = list(Dataset[:, F_idx])
        for attribute_x in set(Attribute_list):
            class_x = Dataset[np.where(Dataset[:, F_idx] == attribute_x)]
            weight = Attribute_list.count(attribute_x) / len(Attribute_list)
            EntropySum += weight * self.InForMationEntropy(class_x)
            Iv_x -= weight * math.log(weight, 2)
        if way_i == 2:
            return (Entropy - EntropySum) / Iv_x
        else:
            return Entropy - EntropySum

    def InformationGini(self, Dataset): #计算数据集基尼系数
        Gini_D = 1
        category_list = list(Dataset[:, -1])
        for category in set(category_list):
            Pm = category_list.count(category) / len(Dataset)
            Gini_D -= Pm**2
        return Gini_D
        pass
    def InformatinoAttributeGini(self, Dataset, f_idx): #计算某一属性的基尼系数
        Gini_A = 0.0
        Attribute_lsit = list(Dataset[:, f_idx])
        for attribute_x in set(Attribute_lsit):
            class_x = Dataset[np.where(Dataset[:, f_idx] == attribute_x)]
            weight = Attribute_lsit.count(attribute_x) / len(Attribute_lsit)
            Gini_A += weight * self.InformationGini(class_x)
        return Gini_A
        pass

    def FindGiniIdx(self,Dataset): #用基尼系数寻找最优属性下标
        best_idx =None
        best_Gini = 1.0 #最优的基尼系数，先设置为1（因为基尼系数<=1）
        for i in range(Dataset.shape[1] - 1):
            present_Gini = self.InformatinoAttributeGini(Dataset, i)#当前属性基尼系数
            if present_Gini < best_Gini:
                best_Gini = present_Gini
                best_idx = i
        return best_idx

    def FindBestIdx(self, Dataset: np.ndarray, way_i):#寻找划分的最优属性下标
        best_idx = 0
        min_gain = 0.0
        best_gain = 0.0 #最大信息增益率,先设置为无穷小
        for i in range(Dataset.shape[1] - 1):
            present_gain = self.InformationGainRate(Dataset, i, way_i) #当前属性信息增益率
            if present_gain > best_gain and math.fabs(present_gain - best_gain) > min_gain:
                best_gain = present_gain
                best_idx = i
        return best_idx

    def GiniBestThreshold(self, Dataset, Idx):#寻找基尼系数下的最优划分阈值
        attribute_list = sorted(list(set(Dataset[:, Idx])))
        thresholds = []
        if len(set(attribute_list)) == 1:
            thresholds = [attribute_list[0]]
        else:
            for i in range(len(attribute_list) - 1):
                thresholds.append((attribute_list[i] + attribute_list[i + 1]) / 2)
        best_gini = 1
        best_threshold = None
        for threshold in thresholds:
            L_threshold = []
            R_threshold = []
            for (idx, d) in enumerate(Dataset[:, Idx]):
                if d < threshold:
                    L_threshold.append(idx)
                else:
                    R_threshold.append(idx)
            L_threshold = Dataset[L_threshold]
            R_threshold = Dataset[R_threshold]
            L_weight = len(L_threshold) / len(Dataset)
            R_wegiht = len(R_threshold) / len(Dataset)
            present_Gini = L_weight * self.InformationGini(L_threshold) + \
                           R_wegiht * self.InformationGini(R_threshold)
            if best_gini > present_Gini:
                best_gini = present_Gini
                best_threshold = threshold
        return best_threshold
        pass

    def BestThreShold(self, Dataset: np.ndarray, Idx): #寻找最优阈值
        Entropy = self.InForMationEntropy(Dataset) #数据集信息熵
        attribute_list = sorted(list(set(Dataset[:, Idx])))
        thresholds = []
        if len(set(attribute_list)) == 1:
            thresholds = [attribute_list[0]]
        else:
            for i in range(len(attribute_list) - 1):
                #thresholds = [(attribute_list[i] + attribute_list[i + 1]) / 2]
                thresholds.append((attribute_list[i] + attribute_list[i + 1]) / 2)
        best_gain = -math.inf #最大信息增益
        best_threshold = None #最优阈值
        for threshold in thresholds:
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
        return best_threshold

    def FindMostClass(self,Class):#寻找类别最多的类
        Class_list = list(Class)
        Class_set = set(Class_list)
        max_Class = None
        Count = 0
        for i in Class_set:
            if Class_list.count(i) > Count:
                Count = Class_list.count(i)
                max_Class = i
        return max_Class


    def TreeNodeGenerate(self, Dataset: np.ndarray, way_i): #建树过程
        Node = None
        if len(set(Dataset[:, -1])) == 1:
            Node = DecisionNode(value=list(set(Dataset[:, -1]))[0])
        elif Dataset.shape[1] == 1:
            Node = DecisionNode(value=self.FindMostClass(Dataset[:, -1]))
        else:
            if way_i == 3:
                best_idx = self.FindGiniIdx(Dataset)
                best_threshold = self.GiniBestThreshold(Dataset, best_idx)
            else:
                best_idx = self.FindBestIdx(Dataset, way_i) #获取划分最优属性的下标
                best_threshold = self.BestThreShold(Dataset, best_idx) #获取划分最优阈值
            Node = DecisionNode(best_idx, best_threshold)
            L_data = Dataset[np.where(Dataset[:, best_idx] < best_threshold)]
            R_data = Dataset[np.where(Dataset[:, best_idx] >= best_threshold)]
            if len(L_data) == 0:
                MostClass = self.FindMostClass(Dataset[:, -1])
                Node.L = DecisionNode(value = MostClass)
            else:
                L_data = np.delete(L_data, best_idx, axis=1)
                Node.L = self.TreeNodeGenerate(L_data, way_i)
            if len(R_data) == 0:
                MostClass = self.FindMostClass(Dataset[:, -1])
                Node.R = DecisionNode(value = MostClass)
            else:
                R_data = np.delete(R_data, best_idx, axis=1)
                Node.R = self.TreeNodeGenerate(R_data, way_i)
        return Node

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target.reshape((len(X), 1))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    print("------------init information-------------")
    print("the shape of X", X_train.shape)
    print("the shape of y", y_train.shape)
    print("the type of X", type(X))
    print("the type of y", type(y))

    print("---------------train log-----------------")

    #信息增益树-----------------------------------------------------------------------------
    Tree_Gain = SimpleDecisionTree()
    #way_i = eval(input("请输入你想要选择的方法(输入数字)： 1. 信息增益 ; 2.信息增益率 ; 3.基尼系数： "))
    Tree_Gain.fit(X, y, 1)
    predict_Gain = Tree_Gain.predict(X_test)
    corr_rate_Gain = (sum([y_test[i] == predict_Gain[i] for i in range(len(y_test))]) / len(y_test))[0]

    #信息增益率树-----------------------------------------------------------------------------
    Tree_Gainrate = SimpleDecisionTree()
    Tree_Gainrate.fit(X, y, 2)
    predict_Gainrate = Tree_Gainrate.predict(X_test)
    corr_rate_Gainrate = (sum([y_test[i] == predict_Gainrate[i] for i in range(len(y_test))]) / len(y_test))[0]

    #基尼系数树-----------------------------------------------------------------------------
    Tree_Gini = SimpleDecisionTree()
    Tree_Gini.fit(X, y, 3)
    predict_Gini = Tree_Gini.predict(X_test)
    corr_rate_Gini = (sum([y_test[i] == predict_Gini[i] for i in range(len(y_test))]) / len(y_test))[0]

    #sklearn树------------------------------------------------------------------------------
    Tree = tree.DecisionTreeClassifier(criterion='entropy')         # sk-learn的决策树模型
    Tree = Tree.fit(X_train, y_train)
    predict_sklearn = Tree.predict(X_test)
    corr_rate_sklearn = (sum([y_test[i] == predict_sklearn[i] for i in range(len(y_test))]) / len(y_test))[0]

    print("---------------the result----------------")
    print("the Entropy rate of correction is", corr_rate_Gain)
    print("the Gainrate rate of correction is", corr_rate_Gainrate)
    print("the Gini rate of correction is", corr_rate_Gini)
    print("the Sklearn rate of correction is", corr_rate_sklearn)
    # predict = Tree.predict(X_test)
    # print("预测：", list(predict))
    # print("实际：", list(y_test[:, 0]))
    #corr_rate = (sum([y_test[i] == predict[i] for i in range(len(y_test))]) / len(y_test))[0]
    #print("the rate of correction is", corr_rate)

    #手动合并列表--------------------------------------------------------
    # for i in range(len(predict_Gini)):
    #     predict_temp = []
    #     predict_temp.append(predict_Gain[i])
    #     predict_temp.append(predict_Gainrate[i])
    #     predict_temp.append(predict_Gini[i])
    #     predict_temp.append(y_test[i])
    #     predict.append(predict_temp)
    #-----------------------------------------------------------------
    data = np.column_stack((predict_Gain, predict_Gainrate))
    data = np.column_stack((data, predict_Gini))
    data = np.column_stack((data, predict_sklearn))
    data = np.column_stack((data, y_test))
    text_meter = ["Entropy",    "GainRate", "Gini","sklearn","reality"]#设置表头
    table = Texttable()
    table.header(text_meter)
    table.set_cols_align(["c", "c", "c","c","c"])
    table.set_cols_valign(['m', 'm', 'm','m','m'])
    table.add_rows(data, header=False)
    print(table.draw())



