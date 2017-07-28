# -*- coding:utf-8 -*-

from numpy import *

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

def smoSimple(dataMatIn, classLabels, C, toler, maxitera):
	dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose() # 100*1
	b = 0; m,n = shape(dataMatrix)
	alphas = mat(zeros((m,1))) # 100*1
	itera = 0
	while (itera < maxitera):
		alphaPairsChanged = 0
		for i in range(m):
			# multiply(alphas,labelMat).T : sum(alpha(i)*yi) 1*100
			# dataMatrix*dataMatrix[i,:].T : 100 *1
			fXi = float(multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
			Ei = fXi - float(labelMat[i])
			if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or \
						((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
				j = selectJrand(i,m)
				fXj = float(multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
				Ej = fXj - float(labelMat[j])
				alphaIold = alphas[i].copy();
				alphaJold = alphas[j].copy();
				if (labelMat[i] != labelMat[j]):
					L = max(0, alphas[j] - alphas[i])
					H = min(C, C + alphas[j] - alphas[i])
				else:
					L = max(0, alphas[j] + alphas[i] - C)
					H = min(C, alphas[j] + alphas[i])
				if L==H: 
					print "L==H"; 
					continue
				eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
				if eta >= 0: print "eta>=0"; continue
				alphas[j] -= labelMat[j]*(Ei - Ej)/eta
				alphas[j] = clipAlpha(alphas[j],H,L)
				if (abs(alphas[j] - alphaJold) < 0.00001): 
					# print "j not moving enough"; 
					continue
				alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
				b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - \
					labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
				b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - \
					labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
				if (0 < alphas[i]) and (C > alphas[i]): 
					b = b1
				elif (0 < alphas[j]) and (C > alphas[j]): 
					b = b2
				else: 
					b = (b1 + b2)/2.0
				alphaPairsChanged += 1
				print "itera: %d i:%d, pairs changed %d" % (itera,i,alphaPairsChanged)
		if (alphaPairsChanged == 0): 
			itera += 1
		else: 
			itera = 0
		print "iteraation number: %d" % itera
	return b,alphas

if __name__ == "__main__":
	dataArr,labelArr = loadDataSet('testSet.txt')
	a = mat([[3],[4],[5]])
	b = mat([[2],[3],[4]])
	print multiply(a,b).T
	# print dataArr
	# b,alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
