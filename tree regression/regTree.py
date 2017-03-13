#!/bin/python

from numpy import *
import pdb

def loadDataSet(file):
	dataMat = []
	fr = open(file)
	for line in fr.readlines():
		curline = line.strip().split("\t")
		floatline = map(float, curline)
		dataMat.append(floatline)
	return dataMat

def binSplitDataSet(dataSet, feature, value):
	mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]
	mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
	return mat0, mat1

def regLeaf(dataSet):
	return mean(dataSet[:, -1])

def regErr(dataSet):
	return var(dataSet[:, -1]) * shape(dataSet)[0]

def createTree(dataSet, leafType= regLeaf, errType=regErr, ops=(1,4)):
	feature, val = chooseBestSplit(dataSet, leafType, errType, ops)
	if feature == None:
		return val
	retTree = {}
	retTree["feat"] = feature
	retTree["val"] = val
	left, right = binSplitDataSet(dataSet, feature, val)
	retTree["left"] = createTree(left, leafType, errType, ops)
	retTree["right"] = createTree(right, leafType, errType, ops)
	return retTree




"""
ops(a, b)-- a:limit the minimal error b:limit the number of leaf
"""
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	if len(set(dataSet[:, -1].T.tolist()[0])) == 1: # when there is a only one entry
		return None, leafType(dataSet)
	m,n = shape(dataSet)
	Err = regErr(dataSet)
	bestFeat = 0
	bestVal = 0
	bestE = inf
	for feat in range(n-1):
		for val in set(dataSet[:, feat]):
			mat0, mat1 = binSplitDataSet(dataSet, feat, val)
			if(shape(mat0)[0] < ops[1] or shape(mat1)[0] < ops[1]): # limit the number of leaf
				continue
			newE = regErr(mat0) + regErr(mat1)
			if newE < bestE:
				bestE = newE
				bestFeat = feat
				bestVal = val
	if Err - bestE < ops[0]:
		return None, leafType(dataSet)
	mat0, mat1 = binSplitDataSet(dataSet, bestFeat, bestVal)
	if(shape(mat0)[0] < ops[1] or shape(mat1)[0] < ops[1]): # limit the number of leaf
		return None, leafType(dataSet)
	return bestFeat, bestVal

def isTree(obj):
	return (type(obj).__name__ == 'dict')


def getMean(tree):
	if isTree(tree['right']) : tree['right'] = getMean(tree['right'])
	if isTree(tree['left'])  : tree['left'] = getMean(tree['left'])
	return (tree['left'] + tree['right']) / 2.0

"""
SME to verify the cost of no merge and merge
"""

def prune(tree, testData):
	if shape(testData)[0] == 0: return getMean(tree)
	if isTree(tree['left']) or isTree(tree['right']):
		left, right = binSplitDataSet(testData, tree['feat'], tree['val'])
	if isTree(tree['left']):
		tree['left'] = prune(tree['left'], left)
	if isTree(tree['right']):
		tree['right'] = prune(tree['right'], right)
	if not isTree(tree['left']) and not isTree(tree['right']):
		left, right = binSplitDataSet(testData, tree['feat'], tree['val'])
		errBefore = sum(power(left[:, -1] - tree['left'], 2)) + sum(power(right[:, -1] - tree['right'], 2))
		treeMean = tree['left'] + tree['right']
		errMerge = sum(power(testData[:, -1] - treeMean, 2))
		if errMerge < errBefore:
			print "merged"
			return treeMean
		else:
			return tree
	else:
		return tree


"""
model tree 
"""
def linearSolve(dataSet):
	m, n = shape(dataSet)
	X = mat(ones((m,n))); Y = mat(ones((m,1)))
	X[:, 1:n] = mat(dataSet[:, 0:n-1])
	Y = dataSet[:, -1]
	xtx = X.T * X
	if linalg.det(xtx) == 0.0:
		raise NameError('This matrix is singular, cannot do inverse,\n\
		try increasing the second value of ops')
	ws = xtx.I * (X.T * Y)
	# pdb.set_trace()
	return ws, X, Y



# def linearSolve(dataSet):   #helper function used in two places
#     m,n = shape(dataSet)
#     X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
#     X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
#     xTx = X.T*X
#     if linalg.det(xTx) == 0.0:
#         raise NameError('This matrix is singular, cannot do inverse,\n\
#         try increasing the second value of ops')
#     ws = xTx.I * (X.T * Y)
#     pdb.set_trace()
#     return ws,X,Y


def modelLeaf(dataSet):
	ws, X, Y = linearSolve(dataSet)
	return ws

def modelErr(dataSet):
	ws, X, Y = linearSolve(dataSet)
	return sum(power(X.ws - Y, 2))


def regTreeVal(model, data):
	return float(model)

def modelTreeVal(model, data):
	n = shape(data)[1]
	x = mat(ones((1, n+1)))
	x[:, 1:n+1] = data
	# pdb.set_trace()
	return float(x * model)

def treeForecast(tree, data, modelEval = regTreeVal):
	if not isTree(tree): return modelEval(tree, data)
	if data[tree['feat']] > tree['val']:
		return treeForecast(tree['left'], data, modelEval)
	else:
		return treeForecast(tree['right'], data, modelEval)

def createForecast(tree, testData, modelEval = regTreeVal):
	m = len(testData)
	pre = zeros((m, 1))
	# pdb.set_trace()
	for i in range(m):
		pre[i,0] = treeForecast(tree, mat(testData[i]), modelEval)
	return pre


if __name__ == '__main__':
	# dataSet = loadDataSet('ex2.txt')
	# dataSet = mat(dataSet)
	# tree = createTree(dataSet, ops=(0,1))
	# print tree
	# testData = loadDataSet('ex2test.txt')
	# testData = mat(testData)
	# prunetree = prune(tree, testData)
	# print prunetree

	# dataSet = loadDataSet('exp2.txt')
	# dataSet = mat(dataSet)
	# tree = createTree(dataSet, modelLeaf, linearErr, ops=(1,10))
	# print tree

	trainSet = loadDataSet('bikeSpeedVsIq_train.txt')
	trainSet = mat(trainSet)
	testSet = loadDataSet('bikeSpeedVsIq_train.txt')
	testSet = mat(testSet)
	tree = createTree(trainSet, ops=(1,20))
	pre = createForecast(tree, testSet[:,0])
	# print pre
	print "tree:", corrcoef(pre, testSet[:,1],rowvar=0)[0,1]

	tree = createTree(trainSet, modelLeaf, modelErr, ops=(1,20))
	pre = createForecast(tree, testSet[:,0], modelTreeVal)
	# print pre
	print "model tree", corrcoef(pre, testSet[:,1],rowvar=0)[0,1]

