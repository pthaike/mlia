#!/bin/python

from numpy import *
import pdb
import matplotlib.pyplot as plt

def loadData():
	x = matrix([[ 1. , 2.1],
		[ 2. , 1.1],
		[ 1.3, 1. ],
		[ 1. , 1. ],
		[ 2. , 1. ]])
	y = [1.0, 1.0, -1.0, -1.0, 1.0]
	return x, y

def loadDataSet(file):
	feaNum = len(open(file).readline().split('\t'))
	dataMat = []
	label = []
	fr = open(file)
	for line in fr.readlines():
		ls = line.strip().split('\t')
		lineArr = []
		for i in range(feaNum-1):
			lineArr.append(float(ls[i]))
		dataMat.append(lineArr)
		label.append(float(ls[-1]))
	return dataMat, label




def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	ret = ones((shape(dataMatrix)[0], 1))
	if threshIneq == "lt":
		ret[dataMatrix[:,dimen] <= threshVal] = -1
	else:
		ret[dataMatrix[:,dimen] > threshVal] = -1
	return ret


def buildStump(dataArr, classLabels, D):
	x = mat(dataArr)
	y = mat(classLabels).T
	m,n = shape(x)
	numStep = 10.0
	bestStump = {}
	minErr = inf
	bestClassEst = mat(zeros((m,1)))
	for dimen in range(n): # for each dimension
		rangeMin = x[:,dimen].min()
		rangeMax = x[:,dimen].max()
		stepSize = (rangeMax - rangeMin) / numStep
		for step in range(-1, int(numStep) + 1):  # for each split step
			for threshIneq in ["lt", "gt"]: # for each inequation
				threshVal = rangeMin + float(step) * stepSize
				err = mat(ones((m,1)))
				pred = stumpClassify(x, dimen, threshVal, threshIneq)
				# pdb.set_trace()
				err[pred==y] = 0
				weightedErr = D.T * err
				if weightedErr < minErr:
					bestStump['dim'] = dimen
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = threshIneq
					minErr = weightedErr
					bestClassEst = pred.copy()
	return bestStump, bestClassEst, minErr

def adaBoostTrain(x, y, numIt=40):
	weakCls = []
	m, n = shape(x)
	D = mat(ones((m,1)) / m)
	aggClassEst = mat(zeros((m,1)))
	for i in range(numIt):
		bestStump, classEst, err = buildStump(x, y , D)
		# print "D:",D.T
		alpha = float(0.5 * log((1-err) / max(err, 1e-16)))
		bestStump["alpha"] = alpha
		weakCls.append(bestStump)
		# print "classEst: ",classEst.T
		# pdb.set_trace()
		expon = multiply(-1 * alpha * mat(y).T, classEst)
		D = multiply(D, exp(expon))
		D = D / D.sum()
		aggClassEst = aggClassEst + alpha * classEst
		# print "aggClassEst: ",aggClassEst.T
		gx = sign(aggClassEst)
		aggErr = multiply(gx != mat(y).T, ones((m,1)))
		errRate = aggErr.sum() / float(m)
		# print "total error: ",errRate,"\n"
		if errRate == 0.0: break
	return weakCls, aggClassEst


def adaClassifier(x, clc):
	x = mat(x)
	m, n = shape(x)
	aggClassEst = mat(zeros((m,1)))
	for c in clc:
		classEst = stumpClassify(x, c['dim'], c['thresh'], c['ineq'])
		aggClassEst  += c['alpha'] * classEst
		print aggClassEst
	return sign(aggClassEst)


def plotROC(predStrengths, classLabels):
	cur = (1.0, 1.0)
	ySum = 0.0
	numPos = sum(array(classLabels)==1.0)
	yStep = 1 / float(numPos)
	xStep = 1 / float(len(classLabels) - numPos)
	sortedIndex = predStrengths.argsort()
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)
	for index in sortedIndex.tolist()[0]:
		if classLabels[index] == 1.0:
			delX = 0
			delY = yStep
		else:
			delX = xStep
			delY = 0
			ySum += cur[1]
		ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
		cur = (cur[0]-delX, cur[1]-delY)
	ax.plot([0,1], [0,1], 'b--')
	plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
	plt.title('ROC curve for AdaBoost Horse Colic Detection System')
	ax.axis([0,1,0,1])
	plt.show()
	print "the Area Under the Curve is: ",ySum*xStep

if __name__ == "__main__":
	# x, y = loadData()
	# clc = adaBoostTrain(x, y, 9)
	# print adaClassifier([0,0], clc)


	x, y = loadDataSet('horseColicTraining2.txt')
	clc, aggClassEst = adaBoostTrain(x, y, 10)
	plotROC(aggClassEst.T, y)
	# xtest, ytest = loadDataSet('horseColicTest2.txt')
	# pred = adaClassifier(xtest, clc)
	# err = mat(ones((shape(pred)[0], 1)))
	# print err[pred != mat(ytest).T].sum()