# -*- coding:utf-8 -*-

from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim='\t'):
	fr = open(fileName)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	datArr = [map(float,line) for line in stringArr]
	return mat(datArr)

def pca(dataMat, topNfeat=9999999):
	meanVals = mean(dataMat, axis=0)
	meanRemoved = dataMat - meanVals
	covMat = cov(meanRemoved, rowvar=0)
	eigVals,eigVects = linalg.eig(mat(covMat))
	eigValInd = argsort(eigVals)
	eigValInd = eigValInd[:-(topNfeat+1):-1]
	redEigVects = eigVects[:,eigValInd]
	lowDDataMat = meanRemoved * redEigVects
	reconMat = (lowDDataMat * redEigVects.T) + meanVals
	return lowDDataMat, reconMat

def pcaMain():
	dataMat = loadDataSet('testSet.txt')
	# lowDMat, reconMat = pca(dataMat, 1)
	lowDMat, reconMat = pca(dataMat, 2)
	print shape(lowDMat)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataMat[:,0].flatten().A[0], dataMat[:,1].flatten().A[0],marker='^', s=90)
	ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0],marker='o', s=50, c='red')
	plt.show()

####################### semiconductor manufacturing data
def replaceNanWithMean():
	datMat = loadDataSet('secom.data', ' ')
	numFeat = shape(datMat)[1]
	for i in range(numFeat):
		meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
		datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
	return datMat

def semMain():
	dataMat = replaceNanWithMean()
	meanVals = mean(dataMat, axis=0)
	meanRemoved = dataMat - meanVals
	covMat = cov(meanRemoved, rowvar=0)
	eigVals,eigVects = linalg.eig(mat(covMat))
	print eigVals

if __name__ == "__main__":
	# pcaMain()
	semMain()
