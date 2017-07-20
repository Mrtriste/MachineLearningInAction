# -*- coding:utf-8 -*-

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
from os import listdir

def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group,labels

def classify0(inX,dataSet,labels,k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX,(dataSetSize,1)) -dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
		sortedClassCount = sorted(classCount.iteritems(),
			key = operator.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]

def file2matrix(filename):
	fr = open(filename)
	numberOfLines = len(fr.readlines())
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	fr = open(filename)
	index = 0
	for line in fr.readlines():
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals- minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals,(m,1))
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals

########################################date web
def datingClassTest():
	hoRatio = 0.30
	datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
	normMat,ranges,minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		res = classify0(normMat[i,:],normMat[numTestVecs:m,:],
			datingLabels[numTestVecs:m],6)
		# print "the classifier came back with: %d, the real answer is: %d"% (res, datingLabels[i])
		if res != datingLabels[i]:
			errorCount += 1.0
	print "error rate:",errorCount/float(numTestVecs)


################################digit detection
def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)
	testFileList = listdir("testDigits")
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
		res = classify0(vectorUnderTest,trainingMat,hwLabels,4)
		if res!= classNumStr:
			errorCount+=1
	print "error rate:",errorCount/mTest


if __name__ == "__main__":
	# group,labels = createDataSet()
	# print classify0([0,0],group,labels,3)

	# group,labels = file2matrix("datingTestSet2.txt")
	# print group
	# print labels[0:20]

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	
	## figure 2.3
	# ax.scatter(group[:,1],group[:,2])

	## figure 2.4
	# ax.scatter(group[:,0],group[:,1],
	# 	15.0*array(labels),15.0*array(labels))

	# figure 2.5
	# ax.scatter(group[:,0],group[:,1],
	# 	15.0*array(labels),15.0*array(labels))

	# plt.show()

	# print autoNorm(group)[0]
	
	datingClassTest()
	
	# handwritingClassTest()

