from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] #shape查看数组的维数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet #inX在行方向上重复dataSetSize次，再与数据集的元素对应相减
    sqDiffMat = diffMat ** 2 #求分类点与其他点的欧式距离
    sqDistances = sqDiffMat.sum(axis=1) #axis表示多维数组的下标，axis=0代表行，axis=1代表列
    distances = sqDistances ** 0.5;
    sortedDistIndicies = distances.argsort() #得到排序后的索引

    classCount = {}
    for i in range(k):# 选择距离最小的k个点
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1# 存在具有相同节点的情况
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # sorted(iterable, cmp=None, key=None, reverse=False)
    return sortedClassCount[0][0]







