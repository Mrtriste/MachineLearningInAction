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