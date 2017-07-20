# MachineLearningInAction
Source code for the book &lt;Machine Learning In Action>  published by Manning



## Chapter 1

```python
# -*- coding:utf-8 -*-

from numpy import *

print random.rand(4,4),"\n"

randMat = mat(random.rand(4,4))
irandMat = randMat.I # 逆矩阵
print randMat,"\n\n",irandMat,"\n"

a = randMat*irandMat
print a,"\n"
print eye(4),"\n" # 单位矩阵
print a-eye(4),"\n"
```



## Chapter 2  -  KNN

- KNN伪代码

```
For every point in our dataset:
  calculate the distance between inX and the current point
  sort the distances in increasing order
  take k items with lowest distances to inX
  find the majority class among these items
  return the majority class as our prediction for the class of inX
```



- 约会网站

datingTestSet2.txt里的标签无法转为int，用datingTestSet2.txt

标准化

When dealing with values that lie in different ranges, it’s common to normalize them.

Common ranges to normalize them to are 0 to 1 or -1 to 1. 

To scale everything from 0 to 1, you need to apply the following formula:

```
newValue = (oldValue-min)/(max-min)
```

```python
def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals- minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals,(m,1))
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals
```

分类函数：

```python
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
```

完整：

```python
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
```



- 数字识别

```python
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
```



### Chapter 3  -  Decision Tree

The kNN algorithm in chapter 2 did a great job of classifying, but it didn’t lead to
any major insights about the data. One of the best things about decision trees is that
humans can easily understand the data.

- 特点

```
Decision trees
Pros: Computationally cheap to use, easy for humans to understand learned results,
missing values OK, can deal with irrelevant features
Cons: Prone to overfitting
Works with: Numeric values, nominal values
```

- createBranch()伪代码

```
Check if every item in the dataset is in the same class:
  If so 
  	  return the class label
  Else
      find the best feature to split the data
      split the dataset
      create a branch node
      for each split
      call createBranch and add the result to the branch node
      return branch node
```

- calculate the Shannon entropy

$$
H = - \sum_{i=1}^{n}p(x_i)log_{2}p(x_i)
$$

```python
from math import log
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]# 每一行的最后一个数据是标签
        if currentLabel not in labelCounts.keys():
        labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
	return shannonEnt
```

