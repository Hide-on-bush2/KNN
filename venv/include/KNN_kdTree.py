import numpy as np
import pandas as pd
import functools

class node:
    def  __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def printTree(root):
    if root.left != None:
        printTree(root.left)
    print(root.val)
    if root.right != None:
        printTree(root.right)



def createKDTree(dataSet, dimension, max_dimension):
    Size = np.size(dataSet, 0)
    if Size == 0:
        return None

    if dimension >= max_dimension:
        dimension = 0

    # def myCmp(a, b):
    #     if a[dimension] > b[dimension]:
    #         return 1
    #     elif a[dimension] < b[dimension]:
    #         return -1
    #     else:
    #         return 0

    # sorted(dataSet, key=functools.cmp_to_key(myCmp))
    dataSet = sorted(dataSet, key=lambda x:x[dimension])
    # print()
    # print(dataSet)
    # print()
    mid_index = Size//2
    currNode = node(dataSet[mid_index])
    currNode.left = createKDTree(dataSet[0:mid_index], dimension + 1, max_dimension)
    currNode.right = createKDTree(dataSet[mid_index+1:], dimension + 1, max_dimension)

    return currNode

def calculateDifference(a, b):
    Dif = a - b
    Sq_dif = Dif ** 2
    return Sq_dif.sum(axis=0)

def KNN_search(x, root, dimension, max_dimension):
    if root.left == None and root.right == None:
        return root.val

    NearestPoint = None
    mark = 0

    if dimension >= max_dimension:
        dimension = 0

    if x[dimension] < root.val[dimension]:
        NearestPoint = KNN_search(x, root.left, dimension + 1, max_dimension) if root.left != None else root.val
        mark = -1
    elif x[dimension] > root.val[dimension]:
        NearestPoint = KNN_search(x, root.right, dimension + 1, max_dimension) if root.right != None else root.val
        mark = 1

    # difference of x and root
    Dif_root = calculateDifference(x, root.val)

    # difference of x and NearestPoint
    Dif_near = calculateDifference(x, NearestPoint)

    if Dif_root < Dif_near:
        NearestPoint = root.val

    def search_opposite(root, x, currNearest):
        if calculateDifference(root.val, x) < calculateDifference(currNearest, x):
            currNearest = root.val
        if root.left != None:
            currNearest = search_opposite(root.left, x, currNearest)

        if root.right != None:
            currNearest = search_opposite(root.right, x, currNearest)
        return currNearest

    if mark == -1:
        NearestPoint = search_opposite(root.right, x, NearestPoint)
    if mark == 1:
        NearestPoint = search_opposite(root.left, x, NearestPoint)

    return NearestPoint

if __name__ == "__main__":
    dataSet = np.loadtxt('dataTestSet2.txt')
    kd_tree = createKDTree(dataSet, 0, 2)
    printTree(kd_tree)
    print()
    Nearest = KNN_search([6, 5], kd_tree, 0, 2)
    print(Nearest)


