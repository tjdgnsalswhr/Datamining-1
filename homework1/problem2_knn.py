import numpy
import matplotlib


class Myclassifier():
	

	def __init__(self, k, traindata_x, traindata_y, vertical):
		self.k = k
		self.traindata_x = traindata_x
		self.traindata_y = traindata_y
		#self_testdata_x = testdata_x
		#self_testdata_y = testdata_y
		self.vertical = vertical
		self.resultmat = numpy.full((self.vertical,3),0)
	#Training..............

	def learning(self, testdata_x,testdata_y):
		self.testdata_x = testdata_x
		self.testdata_y = testdata_y
		print("This is in problem2-knn")
		##print(traindata_x)
		#print(traindata_y)
		#print(testdata_x)
		#print(testdata_y)	
		#print(len(self.traindata_x))
		#print(len(self.traindata_x[0]))
		#print(self.traindata_x[0])
		#print(self.traindata_x[1])
		#arr = self.traindata_x[0] - self.traindata_x[1]
		#print(arr)
		self.resultmat[1][0] = 123
		print(self.resultmat)
		for i in range(0,self.vertical):
			self.resultmat[i][0] = self.traindata_y[i]
		print(self.resultmat)
		self.Getdistance()

	def Getdistance(self):
		self.GetEuclidian()
	#	self.GetManhattan()
	#	self.LLoop()

	
	def GetEuclidian(self):
		for j in range(0,len(self.testdata_y))
			for i in range(0,self.vertical):
				temparray = self.traindata_x[i] - self.testdata_x[j]
				
				
		

	#def GetManhattan(self):


	#def LLoop(self):


