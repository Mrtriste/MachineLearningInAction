# -*- coding:utf-8 -*-

from numpy import *

######################simple adaboost
def loadSimpData():
	datMat = matrix([[ 1. , 2.1],
		[ 2. , 1.1],
		[ 1.3, 1. ],
		[ 1. , 1. ],
		[ 2. , 1. ]])
	classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
	return datMat,classLabels

# Everything on one side of the threshold is thrown into class -1
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
	retArray = ones((shape(dataMatrix)[0],1))
	if threshIneq == 'lt': # 'lt' means let the data less than threshVal be -1
		retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
	else:
		retArray[dataMatrix[:,dimen] > threshVal] = -1.0
	return retArray

def buildStump(dataArr,classLabels,D):
	dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
	m,n = shape(dataMatrix)
	print 'n:',n
	numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
	minError = inf
	for i in range(n):
		rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
		stepSize = (rangeMax-rangeMin)/numSteps
		# print rangeMin,rangeMax,stepSize
		for j in range(-1,int(numSteps)+1):
			# excute two types: to make errorRate <= 0.5
			for inequal in ['lt', 'gt']: # lessthan greaterthan
				threshVal = (rangeMin + float(j) * stepSize)
				predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
				errArr = mat(ones((m,1)))
				errArr[predictedVals == labelMat] = 0
				weightedError = D.T*errArr
				# print "split: dim %d, thresh %.2f, thresh ineqal:%s, the weighted error is %.3f" %\
				# 		(i, threshVal, inequal, weightedError)
				if weightedError < minError:
					minError = weightedError
					bestClasEst = predictedVals.copy()
					bestStump['dim'] = i
					bestStump['thresh'] = threshVal
					bestStump['ineq'] = inequal
	return bestStump,minError,bestClasEst

def simpleMain():
	datMat,classLabels = loadSimpData()
	D = mat(ones((5,1))/5)
	print buildStump(datMat,classLabels,D)

################################## full adaboost
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
	weakClassArr = []
	m = shape(dataArr)[0]
	D = mat(ones((m,1))/m)
	aggClassEst = mat(zeros((m,1))) # record the result of f_m(x), that is sum(G_i(x))

	# You set the number of iterations to 9. But the algorithm reached a total error of 0 
	# after the third iteration and quit, so you didn’t get to see all nine iterations.
	for i in range(numIt):
		bestStump,error,classEst = buildStump(dataArr,classLabels,D)
		print 'error:',error
		# print "D:",D.T
		# calculate alpha
		alpha = float(0.5*log((1.0-error)/max(error,1e-16))) #don’t have a divide-by-zero error
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)
		# print "classEst: ",classEst.T
		# update weight vector D
		expon = multiply(-1*alpha*mat(classLabels).T,classEst)
		D = multiply(D,exp(expon))
		D = D/D.sum()
		# calculate error rate, the result of f(x_i)
		aggClassEst += alpha*classEst
		# print "aggClassEst: ",aggClassEst.T
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
		errorRate = aggErrors.sum()/m
		print "total error: ",errorRate,"\n"
		if errorRate == 0.0: break
	# return weakClassArr # origin
	return weakClassArr,aggClassEst #ROC

def fullMain():
	datMat,classLabels = loadSimpData()
	print adaBoostTrainDS(datMat,classLabels,9)

################################## classify
def adaClassify(datToClass,classifierArr):
	dataMatrix = mat(datToClass)
	m = shape(dataMatrix)[0]
	aggClassEst = mat(zeros((m,1)))
	for i in range(len(classifierArr)):
		classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
				classifierArr[i]['thresh'],classifierArr[i]['ineq'])
		aggClassEst += classifierArr[i]['alpha']*classEst
		# print aggClassEst
	return sign(aggClassEst)

################################## horse dataset
def loadDataSet(fileName):
	numFeat = len(open(fileName).readline().split('\t'))
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr =[]
		curLine = line.strip().split('\t')
		for i in range(numFeat-1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))
	return dataMat,labelMat

def testMain():
	datArr,labelArr = loadDataSet('horseColicTraining2.txt')
	classifierArray = adaBoostTrainDS(datArr,labelArr,50)
	# print classifierArray
	testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
	totalNum = len(testArr)
	prediction10 = adaClassify(testArr,classifierArray)
	errArr=mat(ones((totalNum,1)))
	errNum = errArr[prediction10!=mat(testLabelArr).T].sum()
	print errNum,totalNum
	print "error rate: %f"%(errNum/float(totalNum))

################################### ROC
def plotROC(predStrengths, classLabels):
	import matplotlib.pyplot as plt
	cur = (1.0,1.0)
	ySum = 0.0
	numPosClas = sum(array(classLabels)==1.0)
	yStep = 1/float(numPosClas)
	xStep = 1/float(len(classLabels)-numPosClas)
	sortedIndicies = predStrengths.argsort()
	fig = plt.figure()
	fig.clf()
	ax = plt.subplot(111)
	for index in sortedIndicies.tolist()[0]:
		if classLabels[index] == 1.0:
			delX = 0; delY = yStep;
		else:
			delX = xStep; delY = 0;
			ySum += cur[1]
		ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
		cur = (cur[0]-delX,cur[1]-delY)
	ax.plot([0,1],[0,1],'b--')
	plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
	plt.title('ROC curve for AdaBoost Horse Colic Detection System')
	ax.axis([0,1,0,1])
	plt.show()
	print "the Area Under the Curve is: ",ySum*xStep

def rocMain():
	datArr,labelArr = loadDataSet('horseColicTraining2.txt')
	classifierArray,aggClassEst = adaBoostTrainDS(datArr,labelArr,10)
	plotROC(aggClassEst.T,labelArr)

if __name__ == "__main__":
	# simpleMain()
	# fullMain()
	# testMain()
	rocMain()
