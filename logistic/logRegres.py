#! /bin/python

from numpy import *

def sigmoid(x):
	return 1.0/(1+exp(-x))


"""
update function w = w + alpha * grad = w + alpha * (y - sigmod(wx))*x
"""
def gradAscent(x, y):
	m, n = shape(x)
	alpha = 0.001
	maxCycles = 500
	w = ones((n,1))
	for i in range(maxCycles):
		w = w + alpha * x.transpose()*(y - sigmoid(x*w))
	return w


"""
update weight with a single instance
"""
def stocGradAscent0(x, y):
	m, n = shape(x)
	alpha = 0.01
	w = ones(n)
	for i in range(m):
		w = w + alpha * (y[i] - sigmoid(sum(x[i]*w))) * x[i]
	return w

"""
improve with random index and study ratio alpha
"""
def stocGradAscent(x, y, numIter = 150):
	m,n = shape(x)
	w = ones(n)
	for i in range(numIter):
		dataIndex = range(m)
		for j in range(m):
			alpha = 4 / (1.0 + j + i) + 0.01
			randIndex = int(random.uniform(0, len(dataIndex)))
			w = w + alpha * (y[randIndex] - sigmoid(sum(x[randIndex]*w))) * x[randIndex]
			del(dataIndex[randIndex])
	return w



# load data
def loadDataSet():
	dataMat = []
	labelMat = []
	fr = open("testSet.txt")
	for line in fr.readlines():
		ls = line.strip().split("\t")
		dataMat.append([1.0, float(ls[0]), float(ls[1])])
		labelMat.append(int(ls[2]))
	return dataMat, labelMat


if __name__ == '__main__':
	dat, label = loadDataSet()
	#print gradAscent(dat, label)
	print stocGradAscent(array(dat), label)