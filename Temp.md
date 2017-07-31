## 矩阵求导
https://en.wikipedia.org/wiki/Matrix_calculus
http://blog.csdn.net/xidianliutingting/article/details/51673207
http://blog.sina.com.cn/s/blog_7959e7ed0100w2b3.html

## SMO
http://www.cnblogs.com/biyeymyhjob/archive/2012/07/17/2591592.html
http://blog.csdn.net/v_july_v/article/details/7624837



### nonzero

输入值：数组或矩阵

返回输入值中非零元素的信息（以矩阵的形式）

这些信息中包括 两个矩阵， 包含了相应维度上非零元素所在的行标号，与列标标号。

例如：a=mat([ [1,0,0],[0,0,0],[0,0,0]])

则 nonzero(a) 返回值为两个矩阵：(matrix([[0]], dtype=int32), matrix([[0]], dtype=int32)) , 



### mat

mat的\*，是矩阵的乘，与array的*不一样

```
a = mat([[4,3],[2,1]])
b = mat([[1,2],[3,4]])
print a*b
--
[[13,20],[5,8]]
```

mat.A

将mat转为array

```
c = mat([1,2,3,4,5])
print type(c.A>2),type(c>2)
print c.A>2
print (c.A>2)*(c.A<4)
---
'ndarray','matrix'
[[False,False,True,True,True]]
[[False,False,True,False,False]]
```

