# 1、以ex5data.txt数据为基础，根据Logistic Regression.ipynb
#
# 中的提示，实现longistic算法（必须自己实现，不能调用其它机器学习框架中已经实现的算法）并建立分类模型。再以iris数据集为基础，用实现的longistic算法建立另外一个分类模型，另并计算该模型精确度和准确性。longistic算法的实现过程如下图所示：
#
# logistic.png
#
# 2、(选做)。文件logistic-regression（按神经网络实现）.ipynb 是按神经网络的形式来实现logistic回归，请仔细阅读并理解这些代码作用，分析logistic回归与神经网络有什么关系。
#


import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.model_selection import train_test_split
from misc import Plot


def normalize(X, axis=-1, order=2):
    """ Normalize the dataset X """
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


class Sigmoid():
    """激活函数"""

    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


def accuracy_score(y_true, y_pred):
    """
    比较预测值和真实值,返回准确率
    """
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


class LogisticRegression():
    """ 逻辑回归分类机.
    Parameters:
    -----------
    learning_rate: float
        在使用梯度下降的时候会使用到
    """

    def __init__(self, learning_rate=.1):
        self.param = None
        self.learning_rate = learning_rate

        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        n_features = np.shape(X)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations=4000):
        self._initialize_parameters(X)
        # Tune parameters for n iterations
        for i in range(n_iterations):
            # Make a new prediction
            y_pred = self.sigmoid(X.dot(self.param))
            # Move against the gradient of the loss function with
            # respect to the parameters to minimize the loss
            self.param -= self.learning_rate * -(y - y_pred).dot(X)  # 使用梯度下降调整参数

    def predict(self, X):
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred


# 读取数据
with open('ex5data.txt', 'r') as file:
    # 获取数据
    data = file.read()
    data = data.split('\n')
    data = [whole.split(',') for whole in data]
    data = [[float(subpart) for subpart in part] for part in data]
    data = np.array(data)

    # 数据清洗
    # X = normalize(data.data[data.target != 0])
    X = data[:, [0, 1]]
    X = normalize(X)
    y = data[:, 2]
    y = y.flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # 训练
    clf = LogisticRegression()  # 逻辑回归的实现
    #
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)
    pass
