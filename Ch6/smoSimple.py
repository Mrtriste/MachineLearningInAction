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

################smoSimple
def smoSimple(dataMatIn, classLabels, C, toler, maxiteraa):
	dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose() # 100*1
	b = 0; m,n = shape(dataMatrix)
	alphas = mat(zeros((m,1))) # 100*1
	iteraa = 0
	# You’ll only stop and exit the while loop when you’ve gone through the entire dataset\
	# maxitera number of times without anything changing.
	while (iteraa < maxiteraa):
		alphaPairsChanged = 0
		for i in range(m):
			# multiply(alphas,labelMat).T : alpha(t)*yt 1*100
			# dataMatrix*dataMatrix[i,:].T : x_t*x_i 100 *1
			fXi = float(multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b #sum(at*yt*K(xt,xi))+b
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
				# this is different from tjxxff, because below is : alphas[j] -= labelMat[j]*(Ei - Ej)/eta, -=
				eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
				if eta >= 0: print "eta>=0"; continue # eta = -|Phi(x1)-Phi(x2)|^2
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
				print "iteraa: %d i:%d, pairs changed %d" % (iteraa,i,alphaPairsChanged)
		if (alphaPairsChanged == 0): 
			iteraa += 1
		else: 
			iteraa = 0
		print "iteraaation number: %d" % iteraa
	return b,alphas

def showDataMain():
	dataArr,labelArr = loadDataSet('testSet.txt')
	fig = plt.figure()
	ax = fig.add_subplot(111)
	labelArr = [i+2 for i in labelArr]
	dataArr = array(dataArr)
	ax.scatter(dataArr[:,0],dataArr[:,1],array(labelArr)*15,array(labelArr)*15)
	plt.show()

def smoSimpleMain():
	dataArr,labelArr = loadDataSet('testSet.txt')
	a = mat([[3],[4],[5]])
	b = mat([[2],[3],[4]])
	print multiply(a,b).T
	print dataArr
	b,alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
	print alphas[alphas>0]
	print b


if __name__ == "__main__":
	smoSimpleMain()
	showDataMain()

