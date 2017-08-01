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



### array的copy()

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



### Chapter6  -  SVM



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
ROC竖轴：真阳率/召回率：TP/(TP+FN).预测对的正例/真正的总正例
命中率： TP/(TP + TN) 判对样本中的正样本率
ACC = (TP + TN) / P+N 判对准确率
```

在理想的情况下，最佳的分类器应该尽可能地处手左上角，这就意味着分类器在假阳率很低
的同时获得了很高的真阳率。例如在垃圾邮件的过滤中，这就相当于过滤了所有的垃圾邮件，但没有将任何合法邮件误识为垃圾邮件而放入垃圾邮件的文件夹中。

ROC曲线由两个变量1-specificity 和 Sensitivity绘制. 1-specificity=FPR，即假正类率。Sensitivity即是真正类率，TPR(True positive rate),反映了正类覆盖程度。这个组合以1-specificity对sensitivity,即是以代价(costs)对收益(benefits)。



