# -*- coding:utf-8 -*-

from numpy import *

class treeNode:
	def __init__(self, nameValue, numOccur, parentNode):
		self.name = nameValue
		self.count = numOccur
		self.nodeLink = None
		self.parent = parentNode
		self.children = {}
	def inc(self, numOccur):
		self.count += numOccur

	def disp(self, ind=1): # DFS
		print ' '*ind, self.name, ' ', self.count
		for child in self.children.values():
			child.disp(ind+1)

def testNodeMain():
	rootNode = treeNode('pyramid',9, None)
	rootNode.children['eye']=treeNode('eye', 13, None)
	rootNode.disp()
	rootNode.children['phoenix']=treeNode('phoenix', 3, None)
	rootNode.disp()

#################### create tree
def createTree(dataSet, minSup=1):
	print 'dataSet:',dataSet
	headerTable = {}
	for trans in dataSet:
		for item in trans:
			headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
	# filter some key
	for k in headerTable.keys():
		if headerTable[k] < minSup:
			del(headerTable[k])

	freqItemSet = set(headerTable.keys())
	if len(freqItemSet) == 0: 
		return None, None
	for k in headerTable:
		headerTable[k] = [headerTable[k], None]
	retTree = treeNode('Null Set', 1, None)
	for tranSet, count in dataSet.items():
		localD = {}
		# 将每一行数据去除掉支持度不符合的
		for item in tranSet:
			if item in freqItemSet:
				localD[item] = headerTable[item][0] #value第一个数据为出现次数
		if len(localD) > 0:
			orderedItems = [v[0] for v in sorted(localD.items(),key=lambda p: p[1], reverse=True)]
			updateTree(orderedItems, retTree, headerTable, count)
	return retTree, headerTable

# 将items的第一个数据加到合适的节点，count是整行数据出现了多少次
def updateTree(items, inTree, headerTable, count):
	if items[0] in inTree.children:
		inTree.children[items[0]].inc(count)
	else:
		inTree.children[items[0]] = treeNode(items[0], count, inTree)
		if headerTable[items[0]][1] == None:
			headerTable[items[0]][1] = inTree.children[items[0]]
		else:
			# 将当前数据加到头指针表里，headTable的value中第一个数据是次数
			updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
	if len(items) > 1:
		updateTree(items[1::], inTree.children[items[0]],headerTable, count)

def updateHeader(nodeToTest, targetNode):
	while (nodeToTest.nodeLink != None):# 跳到最后一个节点
		nodeToTest = nodeToTest.nodeLink
	nodeToTest.nodeLink = targetNode

def loadSimpDat():
	simpDat = [['r', 'z', 'h', 'j', 'p'],
				['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
				['z'],
				['r', 'x', 'n', 'o', 's'],
				['y', 'r', 'x', 'z', 'q', 't', 'p'],
				['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
	return simpDat

def createInitSet(dataSet):
	retDict = {}
	for trans in dataSet:
		retDict[frozenset(trans)] = 1
	return retDict

def createTreeMain():
	simpDat = loadSimpDat()
	initSet = createInitSet(simpDat)
	myFPtree, myHeaderTab = createTree(initSet, 3)
	myFPtree.disp()

#########################find prefix path
def ascendTree(leafNode, prefixPath):
	if leafNode.parent != None:
		prefixPath.append(leafNode.name)
		ascendTree(leafNode.parent, prefixPath)

# 将以treenode结尾的路径全部找出来
def findPrefixPath(basePat, treeNode):
	condPats = {}
	while treeNode != None:
		prefixPath = []
		ascendTree(treeNode, prefixPath)
		if len(prefixPath) > 1:
			condPats[frozenset(prefixPath[1:])] = treeNode.count
		treeNode = treeNode.nodeLink
	return condPats

def findPrePathMain():
	simpDat = loadSimpDat()
	initSet = createInitSet(simpDat)
	myFPtree, myHeaderTab = createTree(initSet, 3)
	print findPrefixPath('x', myHeaderTab['x'][1])
	print findPrefixPath('z', myHeaderTab['z'][1])
	print findPrefixPath('r', myHeaderTab['r'][1])

##################### condition tree
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
	bigL = [v[0] for v in sorted(headerTable.items(),key=lambda p: p[1])]
	print bigL
	for basePat in bigL:
		newFreqSet = preFix.copy()
		print 'new:',newFreqSet
		newFreqSet.add(basePat)
		print 'new1:',newFreqSet
		freqItemList.append(newFreqSet)
		condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
		myCondTree, myHead = createTree(condPattBases,minSup)
		if myHead != None:
			print 'conditional tree for: ',newFreqSet
			myCondTree.disp(1)
			mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def conTreeMain():
	freqItems = []
	simpDat = loadSimpDat()
	initSet = createInitSet(simpDat)
	myFPtree, myHeaderTab = createTree(initSet, 3)
	mineTree(myFPtree, myHeaderTab, 3, set([]), freqItems)
	print freqItems

if __name__ == "__main__":
	# testNodeMain()
	# createTreeMain()
	# findPrePathMain()
	conTreeMain()
