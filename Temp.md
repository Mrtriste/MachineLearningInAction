### Chapter8

##### 最小二乘法( OLS,ordinary least square)有解的条件是X列满秩。

(其实有时候会看到是)

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

[知乎上有个回答](https://www.zhihu.com/question/28221429/answer/50909208)介绍了三总理解方式

1. 从上述计算的角度回答
2. 从优化问题的角度。

误差的公式为$$\sum_{i=1}^m (y_i-x_i^Tw)^2+\lambda\sum_{j=1}^pw_j^2$$ ，与普通的最小二乘法计算的误差不同之处在于后面多了平方和。用矩阵表示为$$(Y-Xw)^2+\lambda w^2$$，对w求导并令为0得：$$-2(X^T(Y-Xw))+2\lambda w=0$$，求得$$w=(X^TX+\lambda I)^{-1}X^TY$$.

通过确定$$\lambda$$的值可以使得在方差和偏差之间达到平衡：随着$$\lambda $$的增大，模型方差减小而偏差增大。

方差指的是模型之间的差异，而偏差指的是模型预测值和数据之间的差异。我们需要找到方差和偏差的折中。方差，是形容数据分散程度的，算是“无监督的”，客观的指标，偏差，形容数据跟我们期望的中心差得有多远，算是“有监督的”，有人的知识参与的指标。

3. 从多变量回归的变量选择来说，普通的多元线性回归要做的是变量的剔除和筛选，而岭回归是一种shrinkage的方法，就是收缩。

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

#####  Lasso

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




















