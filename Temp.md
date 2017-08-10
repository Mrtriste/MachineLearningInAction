### pca

- pca

http://blog.csdn.net/xiaojidan2011/article/details/11595869

http://blog.codinglabs.org/articles/pca-tutorial.html

- 协方差

http://blog.csdn.net/wuhzossibility/article/details/8087863

- 协方差和相关系数

https://www.zhihu.com/question/20852004

- 方差和标准差

https://www.zhihu.com/question/20534502/answer/15411212

标准差和均值的量纲一致，因为平方后又开根

- 方差和协方差除以的是n-1

http://blog.csdn.net/maoersong/article/details/21823397

- python切片

http://www.cnblogs.com/weidiao/p/6428681.html

- python argsort()

http://www.cnblogs.com/yyxf1413/p/6253995.html

rowvar=1 表示行代表维度，也就是一列表示一个样本，否则一行表示一个样本

http://blog.csdn.net/u012162613/article/details/42177327 实现解释

原来的正交基为n\*n矩阵，每一行为一个基，数据为n\*m矩阵，每一列为一个样本

我们希望将n维转换为R维，也就是R\*n的正交基矩阵，那么(R\*n)\*(n\*m)=R\*m，也就是每一个n维样本变成R维的样本。

在Python实现中，原来的正交基为n\*n矩阵，每一列为一个基，数据为m\*n矩阵，每一行为一个样本，转换为R维，也就是n\*R的正交基矩阵(m\*n)\*(n\*R)=(m\*R)



- 基变换

原始基$\alpha_1,\alpha_2,...,\alpha_n$到基$\beta_1,\beta_2,...\beta_n$ ，其中$\beta_i$可以由原始基线性表示，即
$$
\beta_j=\sum_{i=1}^nc_{ij}\alpha_i
$$
用矩阵表示即为$\beta=\alpha C$ ，（其中$\alpha、\beta$是n*n的矩阵，每一列是一个向量）其中C称为过渡矩阵，特殊的，如果$\alpha$为标准正交单位基，那么过渡矩阵就是$\beta$

假设在$\alpha$下的坐标为x，在$\beta$下的坐标为y，那么$x=Cy$或$y=C^{-1}x$。

这里的y不是变换之后留在原始基的空间里的坐标，而是在变换后的基的空间里的坐标。

如果$\beta$为正交矩阵，那么$y=C^{-1}x=\beta^{-1}x=\beta^{T}x$。就是[PCA](http://blog.codinglabs.org/articles/pca-tutorial.html)那一块推导的基变换。但是那边附带的知识是，变换后的基是低维空间的基，也就是基变换的时候顺便降维了。







- SVD

http://blog.csdn.net/acdreamers/article/details/44656963

http://blog.csdn.net/zhongkejingwang/article/details/43053513

http://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html

https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/book-chapter-4.pdf
