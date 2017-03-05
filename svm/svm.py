#! /bin/python

from numpy import *

class optStruct(object):
	"""docstring for optStruct"""
	def __init__(self, x, y, C, toler):
		self.x = x
		self.y = y
		self.C = C
		self.toler = toler
		self.m = shape(x)[0]
		self.alphas = mat(zeros((self.m, 1)))
		self.b = 0
		self.eCache = mat(zeros((self.m, 2)))

def calcEk(oS, k):
	f_xk = float(multiply(oS.alphas, oS.y).T * kernel(oS.x, oS.x[k,:])) + oS.b
	Ek = f_xk - float(oS.y[k])
	return Ek

def selectJ(i, Ei,oS):
	maxDelta = 0
	maxK = -1
	Ej = 0
	oS.eCache[i] = [1, Ei]
	validECacheList = nonzero(oS.eCache[:, 0].A)[0]
	if(len(validECacheList) > 1):
		for k in validECacheList:
			if k == i:
				continue
			Ek = calcEk(oS, k)
			deltaE = abs(Ek -  Ei)
			if deltaE > maxDelta:
				maxDelta = deltaE
				maxK = k
				Ej = Ek
		return maxK, Ej
	else:
		j = selectJrand(i, oS.m)
		Ej = calcEk(oS, j)
	return j, Ej
		
def updateEk(oS, k):
	oS.eCache[k] = calcEk(oS, k)


def innerL(i, oS):
	Ei = calcEk(oS, i)
	if (Ei * oS.y[i] < -oS.toler and oS.alphas[i] < oS.C) or (Ei * oS.y[i] > oS.toler and oS.alphas[i] > 0):
		j, Ej = selectJ(i, Ei, oS)
		alpha_i_old = oS.alphas[i]
		alpha_j_old = oS.alphas[j]
		if oS.y[i] == oS.y[j]:
			L = max(0, alpha_i_old + alpha_j_old - oS.C)
			H = min(oS.C, alpha_j_old + alpha_i_old)
		else:
			L = max(0, alpha_j_old - alpha_i_old)
			H = min(oS.C, oS.C+alpha_j_old - alpha_i_old)
		if L==H:
			return 0
		eta = kernel(oS.x[i], oS.x[i]) + kernel(oS.x[j], oS.x[j]) - 2 * kernel(oS.x[i], oS.x[j])
		if eta <= 0:
			print "eta <= 0"
			return 0
		oS.alphas[j] = alpha_j_old + y[j] * (Ei - Ej) / eta
		oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
		updateEk(oS, j)
		if abs(oS.alphas[j] - alpha_j_old) < 0.00001:
			print "j not moving enough"
			return 0
		oS.alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old- oS.alphas[j])
		updateEk(oS, i)
		b1 = oS.b - Ei - y[i] * kernel(x[i], x[i])*(oS.alphas[i] - alpha_i_old) - y[j] * kernel(x[j], x[i]) * (oS.alphas[j] - alpha_j_old)
		b2 = oS.b - Ej - y[i] * kernel(x[i], x[j]) * (oS.alphas[i] - alpha_i_old) - y[j] * kernel(x[j], x[j]) * (oS.alphas[j] - alpha_j_old)

		if 0 < oS.alphas[i] and oS.alphas[i]  < oS.C:
			oS.b = b1
		elif 0 < oS.alphas[j] and oS.alphas[j]  < oS.C:
			oS.b = b2
		else:
			oS.b = (b1+b2) / 2.0
		return 1
	else:
		return 0

def smoP(x, y, C, toler, maxIter, kTup=('lin', 0)):
	oS = optStruct(mat(x), mat(y).transpose(), C, toler)
	iter = 0
	entireSet = True; alphaPairsChanged = 0
	while (iter < maxIter) and (alphaPairsChanged > 0) or entireSet:
		alphaPairsChanged = 0
		if entireSet:
			for i in range(oS.m):
				alphaPairsChanged += innerL(i, oS)
			iter += 1
		else:
			nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
			iter += 1
		if entireSet: entireSet = False
		elif (alphaPairsChanged == 0): entireSet = True
	return oS.b,oS.alphas



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


def kernel(x, xi):
	return x * xi.T


def smoSimple(x, y, C, toler, maxIter):
	x = mat(x)
	y = mat(y)
	b = 0
	m,n = shape(x)
	alphas = mat(zeros((m,1)))
	iter = 0
	while iter < maxIter:
		alphaChange = 0
		for i in range(m):
			f_xi = float(multiply(alphas, y).T * kernel(x, x[i,:])) + b
			Ei = f_xi - float(y[i])
			if y[i] * f_xi < -toler and alphas[i] < C or\
			y[i] * f_xi > toler and alphas[i] > 0:
				j = selectJrand(i, m)
				f_xj = float(multiply(alphas, y).T * kernel(x, x[j,:])) + b
				Ej = f_xj - y[j]
				alpha_i_old = alpha[i].copy()
				alpha_j_old = alpha[j].copy()
				if y[i] == y[j]:
					L = max(0, alpha_i_old+ alpha_j_old - C)
					H = min(C, alpha_i_old + alpha_j_old)
				else:
					L = max(0, alpha_j_old - alpha_i_old)
					H = min(C + alpha_j_old - alpha_i_old)
				if L==H:
					print "L==H"
					continue
				eta = kernel(x[i], x[i]) + kernel(x[j], x[j]) - 2 * kernel(x[i], x[j])
				alphas[j] += y[i] * (Ei - Ej)
				alphas[j] = clipAlpha(alphas[j], H, L)
				if alphas[j] - alpha_j_old < 0.00001:
					print "j not moving enough"
					continue
				alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alphas[j])
				b1 = b - Ei - y[i] * kernel(x[i], x[i]) * (alphas[i] - alpha_i_old) - y[j] * kernel(x[j], x[i])*(alphas[j] - alpha_j_old)
				b2 = b - Ej - y[i] * kernel(x[i], x[j]) * (alphas[i] - alpha_i_old) - y[j] * kernel(x[j], x[j])*(alphas[j] - alpha_j_old)
				if alphas[i] > 0 and alphas[i] < C:
					b = b1
				elif alphas[j] > 0 and alphas[j] < C:
					b = b2
				else:
					b = (b1 + b2) / 2.0
				alphaChange += 1
				print "iter: %d i:%d, pairs changed %d" % (iter, i, alphaChange)
		if alphaChange == 0:
			iter += 1
		else:
			iter = 0
		print "iteration number: %d" % iter
	return alphas, b


def calWc(alphas, x, y):
	x = mat(x)
	y = mat(y).transpose()
	m,n = shape(x)
	w = zeros((n,1))
	for i in range(m):
		w += multiply(alphas[i] * y[i], x[i].T)
	return w



if __name__ == '__main__':
	x,y = loadDataSet()
	alphas, b = smoP(x,y,0.6,0.001,40)

	print alphas