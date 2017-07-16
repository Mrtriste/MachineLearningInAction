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





