
### Array

http://blog.csdn.net/sunny2038/article/details/9002531

- 同一个NumPy数组中所有元素的类型必须是相同的。
- 属性

```
NumPy数组的维数称为秩（rank），一维数组的秩为1，二维数组的秩为2

在NumPy中，每一个线性的数组称为是一个轴（axes），秩其实是描述轴的数量。比如说，二维数组相当于是两个一维数组，其中第一个一维数组中每个元素又是一个一维数组。所以一维数组就是NumPy中的轴（axes），第一个轴相当于是底层数组，第二个轴是底层数组里的数组。而轴的数量——秩，就是数组的维数。

ndarray.ndim：数组的维数（即数组轴的个数），等于秩。最常见的为二维数组（矩阵）。

ndarray.shape：数组的维度。为一个表示数组在每个维度上大小的整数元组。例如二维数组中，表示数组的“行数”和“列数”。ndarray.shape返回一个元组，这个元组的长度就是维度的数目，即ndim属性。

ndarray.size：数组元素的总个数，等于shape属性中元组元素的乘积。

ndarray.dtype：表示数组中元素类型的对象，可使用标准的python类型创建或指定dtype。另外也可使用前一篇文章中介绍的NumPy提供的数据类型。

ndarray.itemsize：数组中每个元素的字节大小。例如，一个元素类型为float64的数组itemsiz属性值为8(float64占用64个bits，每个字节长度为8，所以64/8，占用8个字节），又如，一个元素类型为complex32的数组item属性为4（32/8）。

ndarray.data：包含实际数组元素的缓冲区，由于一般通过数组的索引获取元素，所以通常不需要使用这个属性。
```

- 方法

```
zeros((3,4)):可创建一个全是0的数组
ones((2,,3,4)):可创建一个全为1的数组
empty((3,4)):创建一个内容随机并且依赖与内存状态的数组。
默认创建的数组类型(dtype)都是float64。
可以用d.dtype.itemsize来查看数组中元素占用的字节数目。
ones((2,3,4),dtype=int16)#手动指定数组中元素类型  
arange():返回一个数列形式的数组 arange(10, 30, 5)  #以10开始，差值为5的等差数列
array**x, 每个元素的x次方
```

- 自定义结构

```
student= dtype({'names':['name', 'age', 'weight'], 'formats':['S32', 'i','f']}, align = True)  

使用dtype函数创建，在第一个参数中，'names'和'formats'不能改变，names中列出的是结构中字段名称，formats中列出的是对应字段的数据类型。S32表示32字节长度的字符串，i表示32位的整数，f表示32位长度的浮点数。最后一个参数为True时，表示要求进行内存对齐。

a= array([(“Zhang”, 32, 65.5), (“Wang”, 24, 55.2)], dtype =student) 
```

- axis

```

  axis=0就是最外层的数组，最外层共有3个数组，每个是个2*2的二维数组
  [[[5 2]
    [4 2]]
   [[1 3]
    [2 3]]
   [[1 1]
    [0 1]]]
  X.sum(axis=0) # 对三个二维数组相加
  array([[7, 6],
         [6, 6]])
  X.sum(axis=1)
  array([[9, 4],
         [3, 6],
         [1, 2]])
  X.sum(axis=2)
  array([[7, 6],
         [4, 5],
         [2, 1]])
```

- 索引

```
  多维数组可以每个轴有一个索引。这些索引由一个逗号分割的元组给出。
  b  
  array([[ 0, 1, 2, 3],  
             [10, 11, 12, 13],  
             [20, 21, 22, 23],  
             [30, 31, 32, 33],  
             [40, 41, 42, 43]])  
  b[2,3]  
  23  
  
  b[0:5, 1] # 每行的第二个元素  
  array([ 1, 11, 21, 31, 41]) 
  
  b[: ,1] # 与前面的效果相同  
  array([ 1, 11, 21, 31, 41])  
  
  b[1:3,: ] # 每列的第二和第三个元素  
  array([[10, 11, 12, 13],  
             [20, 21, 22, 23]])  
```



- 默认

```
当少于提供的索引数目少于轴数时，已给出的数值按秩的顺序复制，确失的索引则默认为是整个切片：
b[-1] # 最后一行，等同于b[-1,:]，-1是第一个轴，而缺失的认为是：，相当于整个切片。
b[i]中括号中的表达式被当作i和一系列:，来代表剩下的轴。NumPy也允许你使用“点”像b[i,...]。
点(...)代表许多产生一个完整的索引元组必要的分号。如果x是秩为5的数组(即它有5个轴)，那么:　　　
  x[1,2,...] 等同于 x[1,2,:,:,:],  
  x[...,3] 等同于 x[:,:,:,:,3]
  x[4,...,5,:] 等同 x[4,:,:,5,:]　
```

- 多维数组的遍历是以是第一个轴为基础的
- 改变shape

```
- a.ravel() # 平坦化数组  
- reshape()的参数是个tuple,可以实现维度提升
  a.reshape((2,3))：有返回值，所谓有返回值，即不对原始多维数组进行修改；
- a.resize(2,3)：无返回值，所谓有返回值，即会对原始多维数组进行修改；
```

- array的copy()

##### python中list的拷贝

```python
list1 = [1,2,3,4,5,6]
list2 = list1 #与list1指向同一内存
list3 = list1[1:4] #产生新list
list4 = list1[:] #产生新list
```

##### numpy中array的拷贝

```python
list1 = array([1,2,3,4,5,6])
list2 = list1
list3 = list1[1:4]
list4 = list1[:]
list5 = list1.copy()
list6 = list1[1:4].copy()
# list1,list2,list3,list4都指向同一块内存，只是视图有可能不同，修改视图内存中的数据不会变，但修改数据的话内存中的数据会变，比如list3[1]=9,数据变成[1,2,9,4,5,6]
# list5和list6将数据复制到新的内存中，有自己的一块内存存储数据。
```



### tile

tile(A,reps)

假定A的维度为d,reps的长度为len

当d>=len时，将reps长度补足为d，即在reps前面加上d-len个1。

将A按照与reps的一一对应的方式copy

```
>>> a=[[1,2],[2,3]]
>>> tile(a,2)
array([[1, 2, 1, 2],
       [2, 3, 2, 3]])
这里a的维度为2，reps长度为1（仅仅是1个int类型数据）
则将reps长度补足为2，结果为reps = [1,2](这里也可以写成reps=(1,2)，都无妨的)
进行copy操作，从低维进行.数组a为a[2][2]
一维copy操作：copy两次。a[0]变为[1,2,1,2],a[1]变为[2,3,2,3]
二维copy操作，copy1次。a变为[[1,2,1,2],[2,3,2,3]]
```



### matrix 与 array

Numpy matrices必须是2维的,但是 numpy arrays (ndarrays) 可以是多维的（1D，2D，3D····ND）. Matrix是Array的一个小的分支，包含于Array。所以matrix 拥有array的所有特性。

- 在numpy中matrix的主要优势是：相对简单的乘法运算符号。例如，a和b是两个matrices，那么a*b，就是矩阵积。相反的是在numpy里面arrays遵从逐个元素的运算.

```
c=np.array([[4, 3],
			[2, 1]])
d=np.array([[1, 2], 
			[3, 4]])
print(c*d)
# [[4 6]
#  [6 4]]
```

而矩阵相乘，则需要numpy里面的dot命令 :

```
print(np.dot(c,d))
# [[13 20]
#  [ 5  8]]
```

- matrix 和 array 都可以通过objects后面加`.T` 得到其转置。但是 matrix objects 还可以在后面加 `.H` f得到共轭矩阵, 加 `.I` 得到逆矩阵。
- `**` 运算符的作用也不一样 ：

```
print(a**2)
# [[22 15]
#  [10  7]]
print(c**2)
# [[16  9]
#  [ 4  1]]
```

因为a是个matrix，所以a\*\*2返回的是a*a，相当于矩阵相乘。而c是array，c\*\*2相当于，c中的元素逐个求平方。

- 在做归约运算时，array的维数会发生变化，但matrix总是保持为2维。例如下面求平均值的运算

```
>>> m = np.mat([[1,2],[2,3]])
>>> m
matrix([[1, 2],
        [2, 3]])
>>> mm = m.mean(1)
>>> mm
matrix([[ 1.5],
        [ 2.5]])
```

对array 来说

```
a = np.array([[1,2],[2,3]])
>>> a
array([[1, 2],
       [2, 3]])
>>> am = a.mean(1)
>>> am.shape
(2,)
>>> am
array([ 1.5,  2.5])
```

- mat的*

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



### nonzero

输入值：数组或矩阵

返回输入值中非零元素的信息（以矩阵的形式）

这些信息中包括 两个矩阵， 包含了相应维度上非零元素所在的行标号，与列标标号。

例如：a=mat([ [1,0,0],[0,0,0],[0,0,0]])

则 nonzero(a) 返回值为两个矩阵：(matrix([[0]], dtype=int32), matrix([[0]], dtype=int32)) , 



### mat()函数

mat()函数可以将数组转化为矩阵，在原有数组上修改



### transpose()

Returns a view of the array with axes transposed.不改变原数组，只返回view

- For a 1-D array, this has no effect. (To change between column and row vectors, first cast the 1-D array into a matrix object.) 

对一维不起作用，除非先转换成matrix,用mat()

- For a 2-D array, this is the usual matrix transpose. For an n-D array, if axes are given, their order indicates how the axes are permuted (see Examples). If axes are not provided and `a.shape = (i[0], i[1], ... i[n-2], i[n-1])`, then`a.transpose().shape = (i[n-1], i[n-2], ... i[1], i[0])`.

无参数的话就是按轴的逆序转换

如果传tuple的话，那么在j位置上的i表示，矩阵a的i轴上的数是a.transpose()j轴上的数

对高维数组，可以用transponse进行更复杂的转置：

```
arr = np.arange(16).reshape((2, 2, 4))
arr

array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7]],

       [[ 8,  9, 10, 11],
        [12, 13, 14, 15]]])

arr.transpone((1, 0, 2))

array([[[ 0,  1,  2,  3],
        [ 8,  9, 10, 11]],

       [[ 4,  5,  6,  7],
        [12, 13, 14, 15]]])
```

这个稍微复杂一点，想明白确实想了一段时间。这个重塑，分为两步：

1. 结构的调整 
     首先输入`arr.shape`，得到数组结构`(2, 2, 4)`，transponse参数`（1， 0， 2）`，也就是说该数组调整后，结构不变还是`(2, 2, 4)`。
2. 索引的改变 
     比如说arr[0, 1, 0] = 4, 转置后, 1和0调换，所以，调整后4的索引为[1, 0, 0],其余的数字以此类推。




### argsort

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




### 通用函数(ufunc)

NumPy提供常见的数学函数如 `sin` , `cos` 和 `exp` 。在NumPy中，这些叫作“通用函数”(ufunc)。在NumPy里这些函数作用按数组的元素运算，产生一个数组作为输出。

```
>>> B = arange(3)
>>> B
array([0, 1, 2])
>>> exp(B)
array([ 1.        ,  2.71828183,  7.3890561 ])
>>> sqrt(B)
array([ 0.        ,  1.        ,  1.41421356])
>>> C = array([2., -1., 4.])
>>> add(B, C)
array([ 2.,  0.,  6.])
```
