# -*- coding:utf-8 -*-

from numpy import *

def loadDataSet():
	return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
	C1 = []
	for transaction in dataSet:
		for item in transaction:
			if not [item] in C1:
				C1.append([item])
	C1.sort()
	'''
	map(func, seq1[, seq2,…]) 
	第一个参数接受一个函数名，后面的参数接受一个或多个可迭代的序列，返回的是一个集合。 
	Python函数编程中的map()函数是将func作用于seq中的每一个元素，并将所有的调用的结果作为一个list返回。
	'''
	return map(frozenset, C1)

# 过滤掉支持度不符合的项
def scanD(D, Ck, minSupport):
	ssCnt = {}
	for tid in D:
		# 用frozenset的作用体现在这里，ssCnt[can],可以作为key
		for can in Ck:
			if can.issubset(tid):
				if not ssCnt.has_key(can): 
					ssCnt[can]=1
				else: 
					ssCnt[can] += 1
	numItems = float(len(D))
	retList = []
	supportData = {}
	for key in ssCnt: # frozenset is the key
		support = ssCnt[key]/numItems
		if support >= minSupport:
			retList.insert(0,key)
		supportData[key] = support
	return retList, supportData

def Main1():
	dataSet=loadDataSet()
	C1=createC1(dataSet)
	D=map(set,dataSet)
	L1,suppData0=scanD(D, C1, 0.5)
	print L1

#根据前一个每项长度为k-1的列表生成每项长k的列表
def aprioriGen(Lk, k): #creates Ck
	retList = []
	lenLk = len(Lk)
	for i in range(lenLk):
		for j in range(i+1, lenLk):
			# 除了最后一个元素的
			L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
			L1.sort(); L2.sort()
			if L1==L2:
				retList.append(Lk[i] | Lk[j])
	return retList

#full apriori
def apriori(dataSet, minSupport = 0.5):
	C1 = createC1(dataSet)
	D = map(set, dataSet)
	L1, supportData = scanD(D, C1, minSupport)
	L = [L1]
	k = 2
	while (len(L[k-2]) > 0):
		#根据前一个每项长度为k-1的列表生成每项长k的列表
		Ck = aprioriGen(L[k-2], k)
		# 过滤掉支持度不满足的项
		Lk, supK = scanD(D, Ck, minSupport)
		supportData.update(supK) # 增加一些键值对
		L.append(Lk)
		k += 1
	return L, supportData

def fullAprioriMain():
	dataSet=loadDataSet()
	L,suppData=apriori(dataSet)
	print L
	L,suppData=apriori(dataSet,0.7)
	print L

#########################Association rule-generation functions
def generateRules(L, supportData, minConf=0.7):
	bigRuleList = []
	# L[0]中的项只有一个元素
	for i in range(1, len(L)):
		for freqSet in L[i]: # freqSet like:frozenset([0,1,2])
			H1 = [frozenset([item]) for item in freqSet]
			if (i > 1): # 每个项多于两个元素
				rulesFromConseq(freqSet, H1, supportData, bigRuleList,minConf)
			else:
				calcConf(freqSet, H1, supportData, bigRuleList, minConf)
			'''
			calcConf(freqSet, H1, supportData, bigRuleList, minConf) # H中只有一个元素
			if (i > 1): # 每个项多于两个元素
				# 大于2个元素的项 H才能大于一个元素
				rulesFromConseq(freqSet, H1, supportData, bigRuleList,minConf) 
			'''
	return bigRuleList

# freqSet是总集合，H是P->H中的H列表
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
	prunedH = []
	# H like: [{0},{1}], freqSet like:frozenset([0,1])
	for conseq in H:
		# print 'sss:',freqSet-conseq,conseq
		# P->Q: support(P|Q)/support(P)
		conf = supportData[freqSet]/supportData[freqSet-conseq]
		if conf >= minConf:
			# print freqSet-conseq,'-->',conseq,'conf:',conf
			brl.append((freqSet-conseq, conseq, conf))
			prunedH.append(conseq)
	return prunedH

'''
我们对支持度满足的项分裂成两部分P和H，找出可信度满足的P->H
freqSet = P+H, 参数H是H的列表
我们希望H先是一个元素，然后是两个元素，最多是len(freqSet)-1

'''
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
	# freqSet like:frozenset([0,1,2]), H like: [{0},{1},{2}],
	m = len(H[0])
	if (len(freqSet) > (m + 1)):
		# 由长为m的元素生成长为m+1的元素
		Hmp1 = aprioriGen(H, m + 1)
		Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
		if (len(Hmp1) > 1): #只有大于一个元素才能组合成更长的元素
			rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

def AssociationMain():
	dataSet=loadDataSet()
	L,suppData=apriori(dataSet)
	print L
	rules=generateRules(L,suppData, minConf=0.5)
	print rules
	# rules=generateRules(L,suppData, minConf=0.5)
	# print rules

############################# poisonous mushrooms
def mushRoom():
	mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
	L,suppData=apriori(mushDatSet, minSupport=0.3)
	for item in L[1]:
		if item.intersection('2'): 
			print item
	for item in L[3]:
		if item.intersection('2'): 
			print item

if __name__ == "__main__":
	# Main1()
	# fullAprioriMain()
	# AssociationMain()
	mushRoom()
