Update

When dealing with values that lie in different ranges, it’s common to normalize them.

Common ranges to normalize them to are 0 to 1 or -1 to 1. 

To scale everything from 0 to 1, you need to apply the following formula:

    newValue = (oldValue-min)/(max-min)



    from os import listdir
    
    return sortedClassCount[0][0]
    
    def autoNorm(dataSet):
    	minVals = dataSet.min(0)
    	maxVals = dataSet.max(0)
    	ranges = maxVals- minVals
    	normDataSet = zeros(shape(dataSet))
    	m = dataSet.shape[0]
    	normDataSet = dataSet - tile(minVals,(m,1))
    	normDataSet = normDataSet/tile(ranges,(m,1))
    	return normDataSet,ranges,minVals
    
    ########################################date web
    def datingClassTest():
    	hoRatio = 0.30
    	datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
    	normMat,ranges,minVals = autoNorm(datingDataMat)
    	m = normMat.shape[0]
    	numTestVecs = int(m*hoRatio)
    	errorCount = 0.0
    	for i in range(numTestVecs):
    		res = classify0(normMat[i,:],normMat[numTestVecs:m,:],
    			datingLabels[numTestVecs:m],4)
    		# print "the classifier came back with: %d, the real answer is: %d"% (res, datingLabels[i])
    		if res != datingLabels[i]:
    			errorCount += 1.0
    	print "error rate:",errorCount/float(numTestVecs)
    
    
    ################################digit detection
    def img2vector(filename):
    	returnVect = zeros((1,1024))
    	fr = open(filename)
    	for i in range(32):
    		lineStr = fr.readline()
    		for j in range(32):
    			returnVect[0,32*i+j] = int(lineStr[j])
    	return returnVect
    
    def handwritingClassTest():
    	hwLabels = []
    	trainingFileList = listdir('trainingDigits')
    	m = len(trainingFileList)
    	trainingMat = zeros((m,1024))
    	for i in range(m):
    		fileNameStr = trainingFileList[i]
    		fileStr = fileNameStr.split('.')[0]
    		classNumStr = int(fileStr.split('_')[0])
    		hwLabels.append(classNumStr)
    		trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)
    	testFileList = listdir("testDigits")
    	errorCount = 0.0
    	mTest = len(testFileList)
    	for i in range(mTest):
    		fileNameStr = testFileList[i]
    		fileStr = fileNameStr.split('.')[0]
    		classNumStr = int(fileStr.split('_')[0])
    		vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
    		res = classify0(vectorUnderTest,trainingMat,hwLabels,4)
    		if res!= classNumStr:
    			errorCount+=1
    	print "error rate:",errorCount/mTest
    
    	## figure 2.5
    	# ax.scatter(group[:,0],group[:,1],
    	# 	15.0*array(labels),15.0*array(labels))
    
    	# plt.show()
    
    	# print autoNorm(group)[0]
    	
    	# datingClassTest()
    	
    	handwritingClassTest()



tile

tile(A,reps)

假定A的维度为d,reps的长度为len

当d>=len时，将reps长度补足为d，即在reps前面加上d-len个1。

将A按照与reps的一一对应的方式copy

    >>> a=[[1,2],[2,3]]
    >>> tile(a,2)
    array([[1, 2, 1, 2],
           [2, 3, 2, 3]])
    这里a的维度为2，reps长度为1（仅仅是1个int类型数据）
    则将reps长度补足为2，结果为reps = [1,2](这里也可以写成reps=(1,2)，都无妨的)
    进行copy操作，从低维进行.数组a为a[2][2]
    一维copy操作：copy两次。a[0]变为[1,2,1,2],a[1]变为[2,3,2,3]
    二维copy操作，copy1次。a变为[[1,2,1,2],[2,3,2,3]]


