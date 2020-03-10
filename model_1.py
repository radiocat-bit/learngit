import sklearn
import numpy as np 
import matplotlib.pyplot as plt 
from testCases import *
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1) # 设置随机数种子

# 加载和查看数据集
X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c = np.squeeze(Y), s = 40, cmap = plt.cm.Spectral)

m = Y.shape[1] # 训练集的数量

print("X的维度： " + str(X.shape))
print("Y的维度： " + str(Y.shape))
print("数据集里的数据有: " + str(m) + " 个")

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
LR_predictions = clf.predict(X.T)
print("逻辑回归的准确性: %d " % float((np.dot(Y, LR_predictions) + 
	np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) + 
	"% " + "(正确标记的数据点所占的百分比）")

plt.show()
