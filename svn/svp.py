#! /bin/python


"""
load dataSet
"""
def loadDataSet():
	fr = open("testSet.txt")
	x = []
	y = []
	for line in fr.readlines():
		ls = line.split("\t")
		x.append([float(ls[0]), float(ls[1])])
		y.append(float(ls[2]))
	return x, y


def selectJrand(i, m):
	j = i
	while j==i:
		j = random.uniform(0,m)
	return j

"""
select alpha between L and H
"""
def clipAlpha(aj, H, L):
	if aj > H:
		aj = H
	if aj < L:
		aj = L
	return aj
