# -*- coding:utf-8 -*-

from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t')) - 1
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr =[]
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat

def standRegres(xArr,yArr):
	xMat = mat(xArr); yMat = mat(yArr).T
	xTx = xMat.T*xMat
	print xMat.T.shape,xMat.shape
	print xTx.shape
	# 计算行列式，行列式为0，则逆矩阵不存在
	if linalg.det(xTx) == 0.0:
		print "This matrix is singular, cannot do inverse"
		return
	ws = xTx.I * (xMat.T*yMat)
	return ws

def main1():
	xArr,yArr=loadDataSet('ex0.txt')
	print xArr[0]
	ws = standRegres(xArr,yArr)
	print ws
	xMat=mat(xArr)
	yMat=mat(yArr)
	# correlation
	yHat = xMat*ws
	print corrcoef(yHat.T, yMat)
	# plot
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0])
	xCopy=xMat.copy()
	xCopy.sort(0)
	yHat=xCopy*ws
	ax.plot(xCopy[:,1],yHat)
	plt.show()

############################# LWLR
def lwlr(testPoint,xArr,yArr,k=1.0):
	xMat = mat(xArr); yMat = mat(yArr).T
	m = shape(xMat)[0]
	weights = mat(eye((m)))
	for j in range(m):
		diffMat = testPoint - xMat[j,:]
		weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
	xTx = xMat.T * (weights * xMat)
	if linalg.det(xTx) == 0.0:
		print "This matrix is singular, cannot do inverse"
		return
	ws = xTx.I * (xMat.T * (weights * yMat))
	return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
	m = shape(testArr)[0]
	yHat = zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i],xArr,yArr,k)
	return yHat

def lwlrMain():
	xArr,yArr = loadDataSet('ex0.txt')
	lwlr(xArr[0],xArr,yArr,1.0)
	yHat = lwlrTest(xArr, xArr, yArr,0.01)
	xMat=mat(xArr)
	srtInd = xMat[:,1].argsort(0)
	print srtInd
	xSort=xMat[srtInd][:,0,:]
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(xSort[:,1],yHat[srtInd])
	ax.scatter(xMat[:,1].flatten().A[0], mat(yArr).T.flatten().A[0] , s=2,c='red')
	plt.show()

if __name__ == "__main__":
	# main1()
	lwlrMain()
