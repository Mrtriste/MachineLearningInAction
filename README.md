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

(3) 提高效率

随机梯度上升-This is known as stochastic gradient ascent. Stochastic gradient ascent is an example of an online learning algorithm. This is known as online because we can incrementally update the classifier as new data comes in rather than all at once.

(4) 个别特征数据缺失

一般有以下处理方法

```
1.Use the feature’s mean value from all the available data.
2.Fill in the unknown with a special value like -1.
3.Ignore the instance.
4.Use a mean value from similar items.
5.Use another machine learning algorithm to predict the value.
```

在LR中，采取的办法是将缺失值设为0，且这种方法没有影响，因为

```
weights = weights + alpha * error * dataMatrix[randIndex]
```

如果某个特征的值为0，那么weights对应的特征的改变也为0，并没有朝着哪个方向迈进，因此选0对训练结果没有影响。



### Chapter6  -  SVM

- SMO

http://www.cnblogs.com/biyeymyhjob/archive/2012/07/17/2591592.html
http://blog.csdn.net/v_july_v/article/details/7624837



### Chapter7  -  AdaBoost

- Bagging与Boosting

```
- Bagging:the data is taken from the original dataset S times to make S new datasets. The datasets are the same size as the original. Each dataset is built by randomly selecting an example from the original with replacement.
- Boosting:Each new classifier is trained based on the performance of those already trained. Boosting makes new classifiers focus on data that was previously misclassified by previous classifiers.
```

- decision stump

A decision stump makes a decision on one feature only. It’s a tree with only one split, so it’s a stump.

- 训练流程

```
1.初始化权值分布为均值 D
2.迭代若干次，每次进行以下操作
	a.根据有权值的训练集学习，得到基本分类器，如决策树桩
	b.计算分类误差率e_m
	c.根据分类误差率计算Gm(x)的系数alpha_m
	d.更新权值分布D
3.得到基本分类器的线性组合
	f(x) = sign(f(x)) = sign(sum_1^m(alpha_m*G_m(x)))
```

- 性能度量指标

|      | 预测+1    | 预测-1    | 合计       |
| ---- | ------- | ------- | -------- |
| 真实+1 | TP      | FN      | 真实的正例数P' |
| 真实-1 | FP      | TN      | 真实的负例数N' |
| 合计   | 预测的正例数P | 预测的负例数N |          |

```
正确率：Precision = TP/(TP+FP).预测对的正例/预测的总正例
召回率：Recall = TP/(TP+FN).预测对的正例/真正的总正例，正样本中的正确率
ROC横轴：假阳率/假警报率：FP/(FP+TN).预测错的正例/真正的总负例， 负样本中的错判率
ROC竖轴：真阳率/召回率：TP/(TP+FN).预测对的正例/真正的总正：
命中率： TP/(TP + TN) 判对样本中的正样本率
ACC = (TP + TN) / P+N 判对准确率
```

在理想的情况下，最佳的分类器应该尽可能地处手左上角，这就意味着分类器在假阳率很低
的同时获得了很高的真阳率。例如在垃圾邮件的过滤中，这就相当于过滤了所有的垃圾邮件，但没有将任何合法邮件误识为垃圾邮件而放入垃圾邮件的文件夹中。

ROC曲线由两个变量1-specificity 和 Sensitivity绘制. 1-specificity=FPR，即假正类率。Sensitivity即是真正类率，TPR(True positive rate),反映了正类覆盖程度。这个组合以1-specificity对sensitivity,即是以代价(costs)对收益(benefits)。

- ROC曲线作图流程

```
1.将预测值按预测强度从小到大排序
2.我们从强度最小的点开始，小于这个点的强度分为负例点，大于这个点的分为正例点，那么对于强度最小的点来说，
	ROC纵轴：正样本的判对率为1.0，所有正例都判对了。
	ROC横轴：负样本的错判率为1.0，因为所有真正是负例的样本都判为正例了。
	所以这个点在ROC曲线中对应点(1.0,1.0)
3.不断移动到下一个强度的实例点，小于这个点的强度分为负例点，大于这个点的分为正例点
	如果当前这个点为正例点，则改变true positive rate,即纵轴，否则改变false positive rate,即横轴
```





### Chapter8  -  regression

- 线性回归

找到一个回归系数向量w，用y = xw来计算预测结果，问题就是如何运用现有的数据集找到最合适的w，一个常用的方法就是找出误差最小的w，如果简单地将误差加减，则正值和负值会抵消，因此选用平方误差。

即 
$$
\sum_{i=1}^m(y_i-x_i^Tw)^2
$$
用矩阵替换掉求和符号即为
$$
(Y-Xw)^T(Y-Xw)
$$
对w求导并令其为0，得到
$$
2X(Y-Xw)=0 即Y=Xw
$$
两边同时乘以$$(X^TX)^{-1}X^T$$ ，得到$$w=(X^TX)^{-1}X^Ty$$

代码实现

```python
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
```

- 局部加权线性回归

  http://blog.csdn.net/hujingshuang/article/details/46274723

目前计算x的预测值，用的是全局的数据，且每个数据的权值一样，但实际是与x越接近的数据参考价值越大，所以有了局部加权，体现在误差公式上就是对训练集中的每个点都乘上一个权值系数，然后求和。权值系数用一个m*m的矩阵来表示，对角线上不为0.

误差公式为$$\sum_1^mw(i,i)(y-\theta^Tx)^2$$

用矩阵表示为
$$
J(\theta)=[W(Y-X\theta)]^T(Y-X\theta)=(Y-X\theta)^TW^T(Y-X\theta) \\=(Y^T-\theta^TX^T)W^T(Y-X\theta)\\=Y^TW^TY-\theta^TX^TW^TY-Y^TW^TX\theta+\theta^TX^TW^TX\theta
$$
对$$\theta$$求偏导并令其为0：
$$
-2X^TW^TY+2X^TW^TX\theta=0\\
X^TWY=X^TWX\theta \\
\theta = (X^TWX)^{-1}X^TWY
$$
流程：

1.计算每个训练点的m*m权值矩阵（只有对角线上不为0），常用的权值计算公式有高斯核，离点越近的点权值越大，高斯核如下：
$$
w(i,i)=exp(\frac{|x^{(i)}-x|}{-2k^2})
$$
2.对每个需要预测的点都用上述的求$$\theta$$ 公式计算出权值向量，然后与预测的点的特征向量相乘即可得到预测值，这样的一个缺点就是对每个预测点都要用到全部数据集来计算一次。

```python
# 对单个点的预测
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
# 计算整个训练集每个点的预测值，用于作图
def lwlrTest(testArr,xArr,yArr,k=1.0):
	m = shape(testArr)[0]
	yHat = zeros(m)
	for i in range(m):
		yHat[i] = lwlr(testArr[i],xArr,yArr,k)
	return yHat
```















