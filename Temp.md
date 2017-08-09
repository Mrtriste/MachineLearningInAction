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



- SVD

http://blog.csdn.net/acdreamers/article/details/44656963

http://blog.csdn.net/zhongkejingwang/article/details/43053513

http://www.cnblogs.com/LeftNotEasy/archive/2011/01/19/svd-and-applications.html

https://www.cs.cmu.edu/~venkatg/teaching/CStheory-infoage/book-chapter-4.pdf



- 矩阵乘法的本质 

https://www.zhihu.com/question/21351965/answer/176777987

可以看到三维空间转到二维空间时，用了(a1,a2),(b1,b2),(c1,c2)三个向量，其实可以省去一个向量的，这也就是降维的方式所在，通过除去一些不必要的基，将数据从高维变成低维。