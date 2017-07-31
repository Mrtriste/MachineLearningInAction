# -*- coding:utf-8 -*-

from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName):
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = line.strip().split('\t')
		dataMat.append([float(lineArr[0]), float(lineArr[1])])
		labelMat.append(float(lineArr[2]))
	return dataMat,labelMat

def selectJrand(i,m):
	j=i
	while (j==i):
		j = int(random.uniform(0,m))
	return j

def clipAlpha(aj,H,L):
	if aj > H:
		aj = H
	if L > aj:
		aj = L
	return aj

def showDataMain():
	dataArr,labelArr = loadDataSet('testSet.txt')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	labelArr = [i+2 for i in labelArr]
	dataArr = array(dataArr)
	ax.scatter(dataArr[:,0],dataArr[:,1],array(labelArr)*15,array(labelArr)*15)
	plt.show()

########################## full PlattSMO
class optStruct:
	def __init__(self,dataMatIn, classLabels, C, toler):
		self.X = dataMatIn
		self.labelMat = classLabels
		self.C = C
		self.tol = toler
		self.m = shape(dataMatIn)[0]
		self.alphas = mat(zeros((self.m,1)))
		self.b = 0
		# the first element of eCache is 0 or 1, to represents if it has been calculated
		self.eCache = mat(zeros((self.m,2))) 

def calcEk(oS, k):
	fXk = float(multiply(oS.alphas,oS.labelMat).T*\
			(oS.X*oS.X[k,:].T)) + oS.b
	Ek = fXk - float(oS.labelMat[k])
	return Ek

def selectJ(i, oS, Ei):
	maxK = -1; maxDeltaE = 0; Ej = 0
	oS.eCache[i] = [1,Ei]
	validEcacheList = nonzero(oS.eCache[:,0].A)[0]
	if (len(validEcacheList)) > 1:
		for k in validEcacheList:
			if k == i: continue
			Ek = calcEk(oS, k)
			deltaE = abs(Ei - Ek)  # abs
			if (deltaE > maxDeltaE):
				maxK = k; maxDeltaE = deltaE; Ej = Ek
		return maxK, Ej
	else:# If this is your first time through the loop, you randomly select an alpha.
		j = selectJrand(i, oS.m)
		Ej = calcEk(oS, j)
	return j, Ej

def updateEk(oS, k):
	Ek = calcEk(oS, k)
	oS.eCache[k] = [1,Ek]

def innerL(i, oS):
	# update Ei after every loop
	Ei = calcEk(oS, i)
	# tongjixuexifangfa p129
	# first, go through the point at the bound line, that's 0<alpha<C
	# second, if the points at the bound satisfy the KKT condition, go through the whole set
	# the detect is carried in an allowed error scale
	if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or\
			((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
		j,Ej = selectJ(i, oS, Ei)
		alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();

		if (oS.labelMat[i] != oS.labelMat[j]):
			L = max(0, oS.alphas[j] - oS.alphas[i])
			H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
		else:
			L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
			H = min(oS.C, oS.alphas[j] + oS.alphas[i])
		if L==H: 
			print "L==H"; return 0
		# update alpha_j & Ej
		eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
		if eta >= 0: 
			print "eta>=0"; return 0
		oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
		oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
		updateEk(oS, j) # in the simple version, Ei Ej is calculated every loop
		if (abs(oS.alphas[j] - alphaJold) < 0.00001):
			print "j not moving enough"; return 0
		# calculate alpha_i according to alpha_j, update Ei
		oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
		updateEk(oS, i)
		# update b
		b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
				oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*\
				(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
		b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*\
				oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*\
				(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
		if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): 
			oS.b = b1
		elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): 
			oS.b = b2
		else: 
			oS.b = (b1 + b2)/2.0
		return 1
	else: 
		return 0

def smoP(dataMatIn, classLabels, C, toler, maxitera, kTup=('lin', 0)):
	oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
	itera = 0
	entireSet = True; alphaPairsChanged = 0
	# You’ll exit from the loop whenever 
	# 1.the number of iteraations exceeds your specified maximum 
	# 2.or you pass through the entire set without changing any alpha pairs.
	while (itera < maxitera) and ((alphaPairsChanged > 0) or (entireSet)):
		alphaPairsChanged = 0
		if entireSet:
			for i in range(oS.m):
				alphaPairsChanged += innerL(i,oS)
			print "fullSet, itera: %d i:%d, pairs changed %d" %(itera,i,alphaPairsChanged)
			itera += 1
		else:
			# find points which is support vector
			# that is 0<alpha<C,alpha isn't at the bound,instead of points
			nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
			for i in nonBoundIs:
				alphaPairsChanged += innerL(i,oS)
				print "non-bound, itera: %d i:%d, pairs changed %d" % (itera,i,alphaPairsChanged)
			itera += 1
		# if we go through the whole set, and no alpha changed
		# entireSet = False, alphaPairsChanged = 0, and we will exit while
		if entireSet: 
			entireSet = False
		elif (alphaPairsChanged == 0):
			entireSet = True
		print "iteraation number: %d" % itera
	return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
	# w = sum(alpha_i*y_i*x_i)
	X = mat(dataArr); labelMat = mat(classLabels).transpose()
	m,n = shape(X)
	w = zeros((n,1))
	for i in range(m):
		w += multiply(alphas[i]*labelMat[i],X[i,:].T)
	return w

def smoPlattMain():
	dataArr,labelArr = loadDataSet('testSet.txt')
	b,alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
	print alphas[alphas>0]
	print b
	w = calcWs(alphas,dataArr,labelArr)
	m = len(dataArr)
	error = 0
	for i in range(m):
		res = mat(dataArr[i])*mat(w)+b
		print res
		if labelArr[i]==-1 and res >=0:
			error += 1
		if labelArr[i]==1 and res<=0:
			error+=1
	print 'error rate:%f'%(error/float(m))


if __name__ == "__main__":
	smoPlattMain()
	showDataMain()

