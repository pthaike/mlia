#!/bin/python

from numpy import *

def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', \
				'problems', 'help', 'please'],
				['maybe', 'not', 'take', 'him', \
				'to', 'dog', 'park', 'stupid'],
				['my', 'dalmation', 'is', 'so', 'cute', \
				'I', 'love', 'him'],
				['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				['mr', 'licks', 'ate', 'my', 'steak', 'how',\
				'to', 'stop', 'him'],
				['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1] #1 is abusive, 0 not
	return postingList,classVec

def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

def setOfWord2Vec(vocabList, inputSet):
	vec = [0] * len(vocabList)
	for w in inputSet:
		if w in vocabList:
			vec[vocabList.index(w)] = 1
		else:
			print "word: %s is not in my vocabulary!" % w
	return vec

def bagOfWord2Vec(vocabList, inputSet):
	vec = [0] * len(vocabList)
	for w in inputSet:
		if w in vocabList:
			vec[vocabList.index(w)] += 1
		else:
			print "word: %s is not in my vocabulary!" % w
	return vec

def data2Vecs(vocabList, trainData):
	vecs = []
	for d in trainData:
		vecs.append(setOfWord2Vec(vocabList, d))
	return vecs



def trainNB0(trainMatrix, trainCategory):
	pNum = 2.0
	nNum = 2.0
	fNum = len(trainMatrix[0])
	trainNum = len(trainMatrix)
	pvec = ones(fNum)
	nvec = ones(fNum)
	for i in range(trainNum):
		if trainCategory[i] == 1:
			pvec += trainMatrix[i]
			pNum += sum(trainMatrix[i])
		else:
			nvec += trainMatrix[i]
			nNum += sum(trainMatrix[i])
	pvec = log(pvec / pNum)
	nvec = log(nvec / nNum)
	pAbusive = sum(trainCategory)  / float(trainNum)
	return pvec, nvec, pAbusive

"""
vec2Classify: the vector that need to be classified
pvec, nvec, 
pC1: positive class possibility
"""
def classifyNB(vec2Classify, pvec, nvec, pC1):
	p1 = sum(vec2Classify * pvec) + log(pC1)
	p0 = sum(vec2Classify * nvec) + log(1 - pC1)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	postingList,classVec= loadDataSet()
	vocabList = createVocabList(postingList)
	vecs = data2Vecs(vocabList, postingList)
	pvec, nvec, pAbusive = trainNB0(vecs, classVec)
	testEntry = ['love', 'my', 'dalmation']
	testvec = setOfWord2Vec(vocabList, testEntry)
	print testEntry,'classified as: ',classifyNB(testvec,pvec,nvec,pAbusive)
	testEntry = ['stupid', 'garbage']
	testvec = setOfWord2Vec(vocabList, testEntry)
	print testEntry,'classified as: ',classifyNB(testvec,pvec,nvec,pAbusive)




if __name__ == "__main__":
	
	testingNB()
	