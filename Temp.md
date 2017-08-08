### Chapter12  -  FP Tree

- 构建FP Tree

FP-Tree感觉和字典树很像啊

python实现

```python
class treeNode:
	def __init__(self, nameValue, numOccur, parentNode):
		self.name = nameValue
		self.count = numOccur
		self.nodeLink = None
		self.parent = parentNode
		self.children = {}

def createTree(dataSet, minSup=1):
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
```

- 寻找频繁项








