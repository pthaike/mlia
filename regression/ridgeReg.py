#!/bin/python
from numpy import *

def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) - 1
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr =[]
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat



def ridgeRegres(xMat, yMat, lamda=0.2):
	demon = xMat.T * xMat - lamda * eye(shape(xMat)[1])
	if linalg.det(demon) == 0.0:
		print "This matrix is singular, cannot do inverse"
		return
	w = demon.I * xMat.T * yMat
	return w

def testRidge(xArr, yArr):
	xMat = mat(xArr)
	yMat = mat(yArr).T
	ymean = mean(yMat, 0)
	yMat = yMat - ymean
	xMean = mean(xMat, 0)
	xVar = var(xMat, 0)
	xMat = (xMat - xMean) / xVar
	numTest = 30
	wMat = zeros((numTest, shape(xMean)[1]))
	for i in range(numTest):
		w = ridgeRegres(xMat, yMat, exp(i-10))
		wMat[i,:] = w.T
	return wMat


if __name__ == '__main__':
	dataMat,labelMat = loadDataSet('abalone.txt')
	print testRidge(dataMat,labelMat )
