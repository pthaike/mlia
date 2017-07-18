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
	meanRemove = dataMat - meanVals
	covMat = cov(meanRemove, rowvar = 0)
	eigVals, eigVec = linalg.eig(mat(covMat))
	eigValInd = argsort(eigVals)
	eigValInd = eigValInd[:-(topNfeat+1):-1]
	redEigVec = eigVec[:, eigValInd]
	lowDDataMat = meanRemove * redEigVec
	reconMat = lowDDataMat * redEigVec.T + meanVals
	return lowDDataMat, reconMat


if __name__ == "__main__":
	datArr = loadDataSet("testSet.txt", '\t')
	lowDDataMat, reconMat = pca(datArr, 1)
	print lowDDataMat, reconMat