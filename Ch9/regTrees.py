# -*- coding:utf-8 -*-

from numpy import *

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float,curLine)
		dataMat.append(fltLine)
	return dataMat

def binSplitDataSet(dataSet, feature, value):
	mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
	mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
	return mat0,mat1

def main1():
	testMat=mat(eye(4))
	mat0,mat1=binSplitDataSet(testMat,1,0.5)
	print mat0
	print mat1

################################# regression
def regLeaf(dataSet):
	return mean(dataSet[:,-1])

def regErr(dataSet):
	# 最小的平方误差就是方差*n
	return var(dataSet[:,-1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	tolS = ops[0]; tolN = ops[1]
	# 全属于一类
	if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
		return None, leafType(dataSet)
	m,n = shape(dataSet)
	S = errType(dataSet)
	bestS = inf; bestIndex = 0; bestValue = 0
	for featIndex in range(n-1):
		# print array(dataSet[:,featIndex].T).tolist()
		for splitVal in set(array(dataSet[:,featIndex].T)[0].tolist()):
			mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
			if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): 
				continue
			# 平方误差
			newS = errType(mat0) + errType(mat1)
			if newS < bestS:
				bestIndex = featIndex
				bestValue = splitVal
				bestS = newS
	# 当新分裂的误差与为分裂的误差小于一个阈值，就不分裂
	if (S - bestS) < tolS:
		return None, leafType(dataSet)
	mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
	if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
		return None, leafType(dataSet)
	return bestIndex,bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
	feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
	if feat == None: return val
	retTree = {}
	retTree['spInd'] = feat
	retTree['spVal'] = val
	lSet, rSet = binSplitDataSet(dataSet, feat, val)
	retTree['left'] = createTree(lSet, leafType, errType, ops)
	retTree['right'] = createTree(rSet, leafType, errType, ops)
	return retTree

def regMain():
	# ex00.txt
	myDat=loadDataSet('ex00.txt')
	myMat = mat(myDat)
	print createTree(myMat)
	# ex0.txt
	myDat1=	loadDataSet('ex0.txt')
	myMat1=mat(myDat1)
	print createTree(myMat1)
	'''result:
	{'spInd': 1,
	 'spVal': matrix([[ 0.39435]]), 
	 'right': {'spInd': 1, 
	 			'spVal':matrix([[ 0.197834]]), 
	 			'right': -0.023838155555555553, 
	 			'left':1.0289583666666664
	 			}, 
	 'left': {'spInd': 1,
	 			'spVal': matrix([[ 0.582002]]),
				'right': 1.9800350714285717, 
				'left': {'spInd': 1, 
						'spVal': matrix([[0.797583]]), 
						'right': 2.9836209534883724, 
						'left': 3.9871632000000004
						}
			}
	}
	'''
	# print createTree(myMat,ops=(0,1))
	myDat2= loadDataSet('ex2.txt')
	myMat2=mat(myDat2)
	print createTree(myMat2,ops=(10000,4))

#### 剪枝
def isTree(obj):
	return (type(obj).__name__=='dict')

def getMean(tree):
	if isTree(tree['right']): tree['right'] = getMean(tree['right'])
	if isTree(tree['left']): tree['left'] = getMean(tree['left'])
	return (tree['left']+tree['right'])/2.0

def prune(tree, testData):
	if shape(testData)[0] == 0: 
		return getMean(tree)
	if (isTree(tree['right']) or isTree(tree['left'])):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'],tree['spVal'])
	if isTree(tree['left']): 
		tree['left'] = prune(tree['left'], lSet)
	if isTree(tree['right']): 
		tree['right'] = prune(tree['right'], rSet)
	if not isTree(tree['left']) and not isTree(tree['right']):
		lSet, rSet = binSplitDataSet(testData, tree['spInd'],tree['spVal'])
		errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
				sum(power(rSet[:,-1] - tree['right'],2))
		treeMean = (tree['left']+tree['right'])/2.0
		errorMerge = sum(power(testData[:,-1] - treeMean,2))
		if errorMerge < errorNoMerge:
			print "merging"
			return treeMean
		else: 
			return tree
	else: return tree

if __name__ == "__main__":
	# main1()
	regMain()
