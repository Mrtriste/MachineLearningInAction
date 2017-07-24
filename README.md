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

- 信息增益

https://www.zhihu.com/question/22104055/answer/67014456

熵：表示随机变量的不确定性。

条件熵：在一个条件下，随机变量的不确定性。

信息增益：熵 - 条件熵在一个条件下，信息不确定性减少的程度！

通俗地讲，X(明天下雨)是一个随机变量，X的熵可以算出来， Y(明天阴天)也是随机变量，在阴天情况下下雨的信息熵我们如果也知道的话（此处需要知道其联合概率分布或是通过数据估计）即是条件熵。两者相减就是信息增益！原来明天下雨例如信息熵是2，条件熵是0.01（因为如果是阴天就下雨的概率很大，信息就少了），这样相减后为1.99，在获得阴天这个信息后，下雨信息不确定性减少了1.99！是很多的！所以信息增益大！也就是说，阴天这个信息对下雨来说是很重要的！

所以在特征选择的时候常常用信息增益，如果IG（信息增益大）的话那么这个特征对于分类来说很关键~~ 决策树就是这样来找特征的！

- 选择最佳分裂特征

```python
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0; bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature
```

- 建立决策树

有两个label，一定要分清，特征label(feature label)和类别label(class label)，

```
对每个dataset维护一个feature_label_list,存放当前dataset剩余的feature label
1.将数据集的n个实例的class收集起来，如果全都一样就返回这个class label
2.如果没有特征来split了，返回当前数据集中class最多的那个class label
3.如果都不是，则找到最佳分裂feature，将这个feature从feature_label_list删除
	按照这个feature的取值集合，分别分裂出若干个dataset，对每个dataset递归create tree
```

```python
def createTree(dataSet,labels):
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(
			splitDataSet(dataSet, bestFeat, value),subLabels)
	return myTree
```



### Chapter 4  -  naïve Bayes

- 利用p(x|c)求p(c|x):

$$
p(c|x) = \frac{p(x|c)p(c)}{p(x)}
$$

- 原理

对N维特征向量w（w1,w2,...,wN）的数据集有k个类别，c1 c2 c3 ...ck，现在想知道一个实例(x,y)的分类

对(x,y)求分别属于k个类别的概率
$$
p(c_i|w) = \frac{p(w|c_i)p(c_i)}{p(w)}
$$
取最大的概率作为类别（其中在对不同类别计算的时候，分母p(w)对每个类别一样，因此可以抵消）

对分子，第二项很简单，第一项的话w的各个维度条件独立，所以
$$
p(w|c_i) = p(w^{(1)}|c_i)*p(w^{(2)}|c_i)*...*p(w^{(N)}|c_i)=\prod_{j=1}^Np(w^{(j)}|c_i)
$$


（[更多详细理论](https://mrtriste.github.io/2017/03/15/naiveBayes-%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF/)）

- 训练流程

```python
# 我们的目的是通过上面等式中右边部分来计算左边部分，又分母抵消不用计算，所以我们需要计算的是上述的分子
def trainNB0(trainMatrix,trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)
	p0Num = zeros(numWords); p1Num = zeros(numWords)
	p0Denom = 0.0; p1Denom = 0.0
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i]) #注意为什么加的是特征的个数sum()，而不是实例的个数1
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
    # 向量除以一个数，向量里的每一维都除以这个数
	p1Vect = p1Num/p1Denom # p1Vect = log(p1Num/p1Denom)
	p0Vect = p0Num/p0Denom # p0Vect = log(p0Num/p0Denom)
	return p0Vect,p1Vect,pAbusive

##### 多个小数相乘可能下溢，利用ln(a*b) = ln(a)+ln(b).将乘法改为加法，即改为注释

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
	if p1 > p0:
		return 1
	else:
		return 0
```



### Chapter5  -  Log

For the logistic regression classifier we’ll take our features and multiply each one by a weight and then add them up. This result will be put into the sigmoid, and we’ll get a number between 0 and 1. Anything above 0.5 we’ll classify as a 1, and anything below 0.5 we’ll classify as a 0. You can also think of logistic regression as a probability estimate.

- 原理

$$
\sigma(z)=\frac{1}{1+e^{-z}},z = w^Tx\\即z = w_0\cdot x_0+ w_1\cdot x_1+...+ w_n\cdot x_n
$$

其中w是我们需要训练出来的n维权值向量，x是我们输入的n维特征向量。

大于0.5的类别为1，小于的类别为0.

- 实现

(1) 加载数据中有一行

```
dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
```

线性模型是有一个常数项w0，它代表了拟合线的上下浮动

(2) 训练数据

```python
def gradAscent(dataMatIn, classLabels):
  dataMatrix = mat(dataMatIn)
  labelMat = mat(classLabels).transpose()
  m,n = shape(dataMatrix)
  alpha = 0.001
  maxCycles = 500
  weights = ones((n,1))
  for k in range(maxCycles):
    h = sigmoid(dataMatrix*weights)
    error = (labelMat - h)
    weights = weights + alpha * dataMatrix.transpose()* error
  return weights
```

难理解的是倒数第二行，为什么这样就是梯度上升了呢，其实这后面隐藏了一个巨大的推导过程，详细的推导见

http://blog.csdn.net/dongtingzhizi/article/details/15962797

有非常非常详细的推导。