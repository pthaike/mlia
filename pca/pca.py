from numpy import *

"""
remove the mean
compute the covariance matrix
find the eigenvalues and eigenvectors of the covariance matrix
sort the eigenvalues by desc
take top-N eigenvectors
transform the data into the new space created by the top-N eigenvectors
"""

def loadDataSet(file, delim = '\t'):
	fr = open(file)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	datArr = [map(float, line) for line in stringArr]
	return datArr

def pca(dataMat, topNfeat = 9999999):
	meanVals = mean(dataMat, axis = 0)
	meanRevove = dataMat - meanRevove