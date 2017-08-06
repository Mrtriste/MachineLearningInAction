# -*- coding:utf-8 -*-

from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(fileName):
	dataMat = []
	fr = open(fileName)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float,curLine)
		dataMat.append(fltLine)
	return dataMat

def distEclud(vecA, vecB):
	return sqrt(sum(power(vecA - vecB, 2)))

######################################### K-Means
# 在数据点的范围内随机找k个质心
def randCent(dataSet, k):
	n = shape(dataSet)[1] # n features
	centroids = mat(zeros((k,n)))
	for j in range(n):
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j]) - minJ)
		centroids[:,j] = minJ + rangeJ * random.rand(k,1)
	return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2))) # 属于哪个族，距离的平方
	centroids = createCent(dataSet, k) # k个族，k行n列
	clusterChanged = True
	while clusterChanged:
		clusterChanged = False
		# 对每个实例找到该属于哪个族
		for i in range(m):
			minDist = inf; minIndex = -1
			for j in range(k):
				# 两个行向量的距离
				distJI = distMeas(centroids[j,:],dataSet[i,:])
				if distJI < minDist:
					minDist = distJI; minIndex = j
			if clusterAssment[i,0] != minIndex: 
				clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist**2
		print centroids
		for cent in range(k):
			ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
			centroids[cent,:] = mean(ptsInClust, axis=0) # 对每个特征列计算均值
	return centroids, clusterAssment

def kmeansMain():
	datMat=mat(loadDataSet('testSet.txt'))
	myCentroids, clustAssing = kMeans(datMat,4)
	print '---------------'
	print myCentroids
	print clustAssing

############################################ Binary K-means
def biKmeans(dataSet, k, distMeas=distEclud):
	m = shape(dataSet)[0]
	clusterAssment = mat(zeros((m,2)))
	centroid0 = mean(dataSet, axis=0).tolist()[0]
	centList =[centroid0] # 质心列表
	for j in range(m):
		clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
	while (len(centList) < k):
		lowestSSE = inf
		for i in range(len(centList)):
			ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
			centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2 , distMeas)
			# 当前族划分为两个族后，当前族中数据的误差
			sseSplit = sum(splitClustAss[:,1])
			# 其他族的误差
			sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
			print "sseSplit, and notSplit: ",sseSplit,sseNotSplit
			if (sseSplit + sseNotSplit) < lowestSSE:
				bestCentToSplit = i
				bestNewCents = centroidMat # 两个质心
				bestClustAss = splitClustAss.copy() 
				lowestSSE = sseSplit + sseNotSplit
		# 两个质心中的后一个的index是list的长度
		bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
		# 前一个是被分裂的那个index
		bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
		print 'the bestCentToSplit is: ',bestCentToSplit
		print 'the len of bestClustAss is: ', len(bestClustAss)
		# 原来的质心改成2个中的第一个
		centList[bestCentToSplit] = bestNewCents[0,:]
		# 在列表最后加上2个中的第二个
		centList.append(bestNewCents[1,:])
		# 将被分裂的那个族（属于哪个族，距离的平方）重新赋值，一一对应，没有问题
		clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
	for i in range(k):
		centList[i] = centList[i].tolist()[0]
	return mat(centList), clusterAssment

def bikmeansMain():
	datMat3=mat(loadDataSet('testSet2.txt'))
	centList,myNewAssments=biKmeans(datMat3,3)
	print centList

#####################################Yahoo
def distSLC(vecA, vecB):
	a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
	b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi * (vecB[0,0]-vecA[0,0]) /180)
	return arccos(a + b)*6371.0

def clusterClubs(numClust=5):
	datList = []
	for line in open('places.txt').readlines():
		lineArr = line.split('\t')
		datList.append([float(lineArr[4]), float(lineArr[3])])
	datMat = mat(datList)
	myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
	fig = plt.figure()
	rect=[0.1,0.1,0.8,0.8]
	scatterMarkers=['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
	axprops = dict(xticks=[], yticks=[])
	ax0=fig.add_axes(rect, label='ax0', **axprops)
	imgP = plt.imread('Portland.png')
	ax0.imshow(imgP)
	ax1=fig.add_axes(rect, label='ax1', frameon=False)
	for i in range(numClust):
		ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
		markerStyle = scatterMarkers[i % len(scatterMarkers)]
		# 画原始实例点
		ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0],\
				ptsInCurrCluster[:,1].flatten().A[0],\
				marker=markerStyle, s=90)
	# 画质心
	ax1.scatter(myCentroids[:,0].flatten().A[0],myCentroids[:,1].flatten().A[0], marker='+', s=300)
	plt.show()

if __name__ == "__main__":
	# kmeansMain()
	# bikmeansMain()
	clusterClubs()
