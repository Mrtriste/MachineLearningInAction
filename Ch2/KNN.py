# -*- coding:utf-8 -*-

from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator


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


if __name__ == "__main__":
	# group,labels = createDataSet()
	# print classify0([0,0],group,labels,3)

	group,labels = file2matrix("datingTestSet2.txt")
	print group
	print labels[0:20]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	
	## figure 2.3
	# ax.scatter(group[:,1],group[:,2])

	## figure 2.4
	# ax.scatter(group[:,0],group[:,1],
	# 	15.0*array(labels),15.0*array(labels))

	## figure 2.5
	ax.scatter(group[:,0],group[:,1],
		15.0*array(labels),15.0*array(labels))

	plt.show()

