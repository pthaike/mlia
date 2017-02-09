#!/bin/python

from math import log
import operator
import pdb

		

# calculate the entropy
def calcShannonEnt(dataSet):
	numEnt = len(dataSet)
	labelCount = {}
	for featvec in dataSet:
		label = featvec[-1]
		if label not in labelCount.keys():
			labelCount[label] = 0
		labelCount[label] += 1
	shannonEnt = 0.0
	for key in labelCount:
		p = float(labelCount[key]) / numEnt
		shannonEnt = shannonEnt - p * log(p,2)
	return shannonEnt


def splitDataSet(dataSet, axis, value):
	retDataSet=[]
	for vec in dataSet:
		if vec[axis] == value:
			reducedFeatVec = vec[:axis]
			reducedFeatVec.extend(vec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

#choose the best feature to split the dataset
def chooseBestFeature(dataSet):
	numFeature = len(dataSet[0]) - 1
	bestFeature = -1
	oriEnt = calcShannonEnt(dataSet)
	numdat = len(dataSet)
	bestEnt = 0.0
	for i in range(numFeature):
		allvalue = [vec[i] for vec in dataSet]
		uniqueVals = set(allvalue)
		hGain = 0.0
		for val in uniqueVals:
			splitdat = splitDataSet(dataSet, i, val)
			ent = calcShannonEnt(splitdat)
			prob = float(len(splitdat)) / numdat
			hGain += prob * ent
		curEntgain = oriEnt - hGain
		if curEntgain > bestEnt:
			bestEnt = curEntgain
			bestFeature = i
	return bestFeature, bestEnt

#get the class label that has the most entries
def majorityClass(classList):
	cls = {}
	for c in classList:
		if c not in cls:
			cls[c] = 0
		cls[c] += 1
	sortedcls = sorted(cls, key = operator.itemgetter(1), reverse = True)
	return sortedcls[0][0]

#build tree by dataSet and attrs
def createTree(dataSet, attrs):
	labels = [vec[-1] for vec in dataSet]
	if len(labels) == labels.count(labels[0]):
		return labels[0]
	if len(dataSet[0]) == 1:
		return majorityClass(labels)
	bestFeat,_ = chooseBestFeature(dataSet)
	values = [vec[bestFeat] for vec in dataSet]
	uniqueVals = set(values)
	bestFeatLabel = attrs[bestFeat]
	tree = {bestFeatLabel:{}}
	for val in uniqueVals:
		subattr = attrs[:bestFeat]
		subattr.extend(attrs[bestFeat+1:])
		subDat = splitDataSet(dataSet, bestFeat, val)
		tree[bestFeatLabel][val] = createTree(subDat, subattr)
	return tree

def classify(tree, attrs, x):
	firstStr = tree.keys()[0]
	dic = tree[firstStr]
	featIndex = attrs.index(firstStr)
	next = dic[x[featIndex]]
	if type(next).__name__ == 'dict':
		classLabel = classify(next, attrs, x)
	else:
		classLabel = next
	return classLabel


def createDataSet():
	dataSet = [
	[1,1,'yes'],
	[1,1,'yes'],
	[1,0,'no'],
	[0,1,'no'],
	[0,1,'no']
	]
	attrs = ['no surfacing', 'flippers']
	return dataSet, attrs

if __name__ == '__main__':
	dataSet, attrs = createDataSet()
	tree =  createTree(dataSet, attrs)
	print tree
	print classify(tree, attrs, [1,0])