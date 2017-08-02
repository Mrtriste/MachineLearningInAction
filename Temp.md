### 矩阵求导

https://en.wikipedia.org/wiki/Matrix_calculus
http://blog.csdn.net/xidianliutingting/article/details/51673207
http://blog.sina.com.cn/s/blog_7959e7ed0100w2b3.html



### 行列式

https://www.zhihu.com/question/36966326/answer/70687817



### 矩阵的秩

矩阵的秩: 用初等行变换将矩阵A化为阶梯形矩阵，则矩阵中非零行的个数就定义为这个矩阵的秩, 记为r（A）

https://www.zhihu.com/question/21605094



### 矩阵的初等行变换

https://wenku.baidu.com/view/0e89ed5b3c1ec5da50e27094.html



### 满秩矩阵

 设A是n阶矩阵（方阵）, 若r（A） = n, 则称A为满秩矩阵。

 

###  argsort

从中可以看出argsort函数返回的是数组值从小到大的索引值

```python
# 一维
x = np.array([3, 1, 2])
np.argsort(x)
# array([1, 2, 0])

# 二维
x = np.array([[0, 3], [2, 2]])
np.argsort(x, axis=0) #按列排序
# array([[0, 1],[1, 0]])
np.argsort(x, axis=1) #按行排序
# array([[0, 1],[0, 1]])
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

目前计算x的预测值，用的是全局的数据，且每个数据的权值一样，但实际是与x越接近的数据参考价值越大，所以有了局部加权，















