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
			if y[i] * f_xi < toler and alphas[i] < C or\
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
				else if alphas[j] > 0 and alphas[j] < C:
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