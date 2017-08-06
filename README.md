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

datingTestSet.txt里的标签无法转为int，用datingTestSet2.txt

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

（实现的是ID3算法，特征选择用的是信息增益，没有进行剪枝）

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



### Chapter5  -  Logistic Regression

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

##### 线性回归

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

##### 局部加权线性回归

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

### 

##### 最小二乘法( OLS,ordinary least square)有解的条件是X列满秩。

https://www.zhihu.com/question/28221429

(其实有时候会看到是不满足列满秩或者特征的相关比较强时，不适合用最小二乘法，不满足列满秩就是逆矩阵不存在，那为什么相关性较强也不适用呢？相关性较强意味着X^TX很小，甚至趋向于0，那么逆矩阵就会很大，那么得到的权值就会很不稳定。简单地可以理解为相关性强时，数据就不能提供足够的信息)

为什么？

假设X的大小为m*n
$$
\theta = (X^TX)^{-1}X^TY
$$
有解也就是X^TX有逆矩阵，也就是X^TX满秩，X^TX的大小为n*n，如果满秩的话就是秩为n。

有以下定理：

- [设m×n实矩阵A的秩为n,证明：矩阵A^TA为正定矩阵.](https://www.zybang.com/question/21898d2dcd032648a96b1392dee1e6b4.html)

- 正定矩阵为什么可逆：正定阵的特征值全大于0，而行列式等于特征值的乘积，因此行列式大于0，可逆。

- [行列式不为0，矩阵可逆](https://www.zybang.com/question/9b6a7724ebf34242a7dd3c3b8f8d6c86.html)

- [矩阵可逆的充要条件](https://www.zybang.com/question/98bb5d3881c8dd9b79c57c7241b60054.html)

  ```
  n阶方阵A可逆
  <=> A非奇异
  <=> |A|≠0
  <=> A可表示成初等矩阵的乘积
  <=> A等价于n阶单位矩阵
  <=> r(A) = n
  <=> A的列(行)向量组线性无关
  <=> 齐次线性方程组AX=0 仅有零解
  <=> 非 齐次线性方程组AX=b 有唯一解
  <=> 任一n维向量可由A的列(或行)向量组线性表示
  <=> A的特征值都不为0
  ```


- 矩阵的列秩和行秩总是相等的，因此它们可以简单地称作矩阵A的秩。通常表示为r(A).
- [行(列)满秩矩阵的一些性质及应用](https://wenku.baidu.com/view/19204b4d7e21af45b307a888.html)

##### 岭回归

用最小二乘法最误差有什么问题？在计算(X^TX)^-1的时候，可能不存在，比如特征数大于样例数，即列数大于行数，是不可能行满秩的，也就是上式不可逆。

- 因此从计算的角度可以加上一个$$\lambda I$$ 使$$w=(X^TX+\lambda I)^{-1}X^TY$$ 可以计算。
- 上面是从运算的角度，有没有更自然的理解方式呢？

[知乎上有个回答](https://www.zhihu.com/question/28221429/answer/50909208)介绍了三种理解方式

1. 从上述计算的角度回答
2. 从优化问题的角度。

误差的公式为$$\sum_{i=1}^m (y_i-x_i^Tw)^2+\lambda\sum_{j=1}^pw_j^2$$ ，与普通的最小二乘法计算的误差不同之处在于后面多了平方和。用矩阵表示为$$(Y-Xw)^2+\lambda w^2$$，对w求导并令为0得：$$-2(X^T(Y-Xw))+2\lambda w=0$$，求得$$w=(X^TX+\lambda I)^{-1}X^TY$$.

通过确定$$\lambda$$的值可以使得在方差和偏差之间达到平衡：随着$$\lambda $$的增大，模型方差减小而偏差增大。

方差指的是模型之间的差异，而偏差指的是模型预测值和数据之间的差异。我们需要找到方差和偏差的折中。方差，是形容数据分散程度的，算是“无监督的”，客观的指标，偏差，形容数据跟我们期望的中心差得有多远，算是“有监督的”，有人的知识参与的指标。

1. 从多变量回归的变量选择来说，普通的多元线性回归要做的是变量的剔除和筛选，而岭回归是一种shrinkage的方法，就是收缩。

岭回归的回归参数有先验分布，而最小二乘对参数没有限制。对参数进行先验分布限制，会使得得到的回归参数取值不会很病态.

求w的代码实现：

```python
def ridgeRegres(xMat,yMat,lam=0.2):
	xTx = xMat.T*xMat
	denom = xTx + eye(shape(xMat)[1])*lam
	if linalg.det(denom) == 0.0:
		print "This matrix is singular, cannot do inverse"
		return
	ws = denom.I * (xMat.T*yMat)
	return ws

#### 在预处理数据的时候要对数据集进行标准化
def ridgeTest(xArr,yArr):
	xMat = mat(xArr); yMat=mat(yArr).T
	yMean = mean(yMat,0)
	yMat = yMat - yMean
	xMeans = mean(xMat,0) # 均值，对每一列求均值
	xVar = var(xMat,0) # 方差
	xMat = (xMat - xMeans)/xVar # 随机变量标准化
	numTestPts = 30
	wMat = zeros((numTestPts,shape(xMat)[1]))
	for i in range(numTestPts):
		ws = ridgeRegres(xMat,yMat,exp(i-10))
		wMat[i,:]=ws.T
	return wMat
```

ridgeTest求了30个不同$$\lambda$$时的权值向量，每个权值向量有八个值，对应8个特征的重要性，做出的图展现的是八条曲线，每条曲线表明当前特征随lambda的变化，权值的变化。

##### Lasso

lasso算法在普通最小二乘法的基础上增加了一个条件，完整写的话就是
$$
\sum_{i=1}^m(y_i-x_i^Tw)^2 \\
s.t.  \sum_{j=1}^p|w_j| \le \lambda
$$
岭回归可以改写成
$$
\sum_{i=1}^m(y_i-x_i^Tw)^2 \\ s.t.  \sum_{j=1}^pw_j^2 \le \lambda
$$
可以看出岭回归与lasso只是一个是平方和受限，一个是绝对值和受限。

不同之处就在于当lambda足够小时，有些权值系数会被迫减到0，有些特征直接被忽略掉，因此能更好的展示出数据的特点。

但求解上述式子的解时，需要二次规划，计算复杂，用一种更简单的算法得到差不多的效果。

##### 前向逐步回归

前向逐步算法是一种贪心算法，每一步都尽可能较小误差，一开始所有的权重设为1，每一步的决策时对某个权重增加或减小一个很小的值。

流程：

```
数据标准化，使其分布满足0均值和单位方差
在每轮迭代过程中：
    设置当前最小误差lowestError为正无穷
    对每个特征：
        增大或缩小：
        	改变一个系数得到一个新的w
        	计算新w下的误差
      		如果误差Error小于当前最小误差lowestError：设置Wbest等于当前的w
        将w设置为新的Wbest
```

python实现：

```python
def stageWise(xArr,yArr,eps=0.01,numIt=100):
	xMat = mat(xArr); yMat=mat(yArr).T
	yMean = mean(yMat,0)
	yMat = yMat - yMean
	xMat = regularize(xMat)
	m,n=shape(xMat)
	ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
	returnMat = zeros((numIt,n))
	for i in range(numIt):
		#print ws.T
		lowestError = inf;
		for j in range(n):
			for sign in [-1,1]:
				wsTest = ws.copy()
				wsTest[j] += eps*sign
				yTest = xMat*wsTest
				rssE = rssError(yMat.A,yTest.A)
				if rssE < lowestError:
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy()
		returnMat[i,:]=ws.T
	return returnMat
```





### Chapter9  -  Tree-based regression

CART是classification and regression tree，分类与回归树，正如名字所说的，它其实有两种树，分类树和回归树。第三章中讲的决策树是ID3决策树，根据信息增益作为特征选择算法。

CART树与前面说的树有什么差别呢？

1.之前的生成树的算法在对某个特征切分的时候，将数据集按这个特征的所有取值分成很多部分，这样的切分速度太快，而CART只进行```二元切分```，对每个特征只切分成两个部分。

2.ID3和C4.5只能处理离散型变量，而CART因为是二元切分，```可以处理连续型变量```，而且只要将特征选择的算法改一下的话既可以生成回归树。

本章讲了回归树和模型树

##### 回归树

- 特征选择

  回归树使用的是平方误差最小法作为特征选择算法。

  其思想是将当前节点的数据按照某个特征在某个切分点分成两类，比如$R_1,R_2$，其对应的类别为$C_1,C_2$，我们的任务就是找到一个切分点使误差最小，那么怎么度量误差呢？这里使用的是平方误差，即
  $$
  min[min\sum_{x_i\in R_1}(y_i-c_1)^2+min\sum_{x_i\in R_2}(y_i-c_2)^2]
  $$
  遍历某个特征可取的s个切分点（对离散型变量，要么等于要么不等于；对连续型变量，<或者>=），选择使上式最小的切分点。

  对每个确定的集合，c1,c2取平均值$\sum_{x_i\in R_1}(y_i-c_1)^2$和$\sum_{x_i\in R_2}(y_i-c_2)^2 $才会最小，这样的话就是求划分为两个集合后，分别对每个集合求方差*实例数，加起来的最小值。

- 剪枝

  简单的剪枝，如果merge后的误差更小就merge


python实现

```python
# 选取最佳分裂特征和值
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

# 创建树
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

#简单的剪枝，如果merge后的误差更小就merge
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
```

##### 模型树

树的叶子节点不是一个数值，而是一个模型的参数，如果叶子节点是线性回归模型，那么叶子节点存的就是权值系数w

python实现

```python
# 叶子节点存放的东西
def modelLeaf(dataSet):
	ws,X,Y = linearSolve(dataSet)
	return ws

#线性模型的误差
def modelErr(dataSet):
	ws,X,Y = linearSolve(dataSet)
	yHat = X * ws
	return sum(power(Y - yHat, 2))

createTree(myMat2, modelLeaf, modelErr,(1,10))
```



### Chapter10  -  k-means clustering

k-均值算法流程：

```
创建k个点作为起始质心（经常是随机选择）
当任意一个点的簇分配结果发生改变时
	对数据集中的每个数据点
		对每个质心
  			计算质心与数据点之间的距离
  		将数据点分配到距其最近的簇
  	对每一个簇，计算簇中所有点的均值并将均值作为质心
```

python实现

```python
def randCent(dataSet, k):
	n = shape(dataSet)[1] # n features
	centroids = mat(zeros((k,n)))
	for j in range(n):
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j]) - minJ)
		centroids[:,j] = minJ + rangeJ * random.rand(k,1)
	return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2))) # 属于哪个族，距离的平方
	centroids = createCent(dataSet, k) # k个族，k行n列
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		# 对每个实例找到该属于哪个族
		for i in range(m):
			minDist = inf; minIndex = -1
			for j in range(k):
				# 两个行向量的距离
				distJI = distMeas(centroids[j,:],dataSet[i,:])
				if distJI < minDist:
					minDist = distJI; minIndex = j
			if clusterAssment[i,0] != minIndex: 
				clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist**2
		print centroids
		for cent in range(k):
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
			centroids[cent,:] = mean(ptsInClust, axis=0) # 对每个特征列计算均值
	return centroids, clusterAssment
```



以上的算法可能会收敛于局部最小值，有一种改进的算法是二分K-均值算法

```
将所有点看成一个襄
当簇数目小于k时
	对于每一个簇
		计算总误差
		在给定的簇上面进行K-均值聚类（k=2)
		计算将该簇一分为二之后的总误差
	选择使得误差最小的那个族进行划分操作
```

python实现

```python
def biKmeans(dataSet, k, distMeas=distEclud):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))
	centroid0 = mean(dataSet, axis=0).tolist()[0]
	centList =[centroid0] # 质心列表
	for j in range(m):
		clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
	while (len(centList) < k):
		lowestSSE = inf
		for i in range(len(centList)):
			ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
			centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2 , distMeas)
			# 当前族划分为两个族后，当前族中数据的误差
			sseSplit = sum(splitClustAss[:,1])
			# 其他族的误差
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
			print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
			if (sseSplit + sseNotSplit) < lowestSSE:
				bestCentToSplit = i
				bestNewCents = centroidMat # 两个质心
				bestClustAss = splitClustAss.copy() 
				lowestSSE = sseSplit + sseNotSplit
		# 两个质心中的后一个的index是list的长度
		bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
		# 前一个是被分裂的那个index
		bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
		print 'the bestCentToSplit is: ',bestCentToSplit
		print 'the len of bestClustAss is: ', len(bestClustAss)
		# 原来的质心改成2个中的第一个
		centList[bestCentToSplit] = bestNewCents[0,:]
		# 在列表最后加上2个中的第二个
		centList.append(bestNewCents[1,:])
		# 将被分裂的那个族（属于哪个族，距离的平方）重新赋值，一一对应，没有问题
		clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
	for i in range(k):
		centList[i] = centList[i].tolist()[0]
	return mat(centList), clusterAssment
```




























