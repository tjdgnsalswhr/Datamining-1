import numpy
import matplotlib
import operator

class Myclassifier():
	

	def __init__(self, k, traindata_x, traindata_y, vertical):
		self.k = k
		self.traindata_x = traindata_x
		self.traindata_y = traindata_y
		#self_testdata_x = testdata_x
		#self_testdata_y = testdata_y
		self.vertical = vertical
		#self.resultmat = numpy.full((len(self.testdata_y),self.vertical,3),0)
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

		self.resultmat = numpy.full((len(self.testdata_y),self.vertical,4),0.0)
		#self.resultmat[0][1][1] = 123
		#print(self.resultmat)
		for j in range(0,len(self.testdata_y)):
			for i in range(0,self.vertical):
				self.resultmat[j][i][0] = self.traindata_y[i]
		print(self.resultmat)
	        	
		self.Getdistance()

	def Getdistance(self):
		self.GetEuclidian()
		self.GetManhattan()
		self.GetInfiniteL()

	
	def GetEuclidian(self):
		for i in range(0,len(self.testdata_y)):
			for j in range(0,len(self.traindata_y)):
				temp = self.traindata_x[j]-self.testdata_x[i]
				temp2 = temp**2
				tempsum = numpy.sum(temp2)
				tempresult = numpy.sqrt(tempsum)
				self.resultmat[i][j][1] = tempresult

		print(self.resultmat)

	def GetManhattan(self):
		#temp10 = self.traindata_x[0] - self.testdata_x[1]
		#temp11 = numpy.abs(temp10)
		#print(numpy.sum(temp11))
		for i in range(0,len(self.testdata_y)):
                        for j in range(0,len(self.traindata_y)):
                                temp = self.traindata_x[j]-self.testdata_x[i]
                                temp2 = numpy.abs(temp)
                                tempsum = numpy.sum(temp2)
                                self.resultmat[i][j][2] = tempsum

		
		print(self.resultmat)


	def GetInfiniteL(self):
		for i in range(0,len(self.testdata_y)):
			for j in range(0,len(self.traindata_y)):
				temp = self.traindata_x[j]-self.testdata_x[i]
				temp2 = numpy.abs(temp)
				tempresult = numpy.max(temp2)
				self.resultmat[i][j][3] = tempresult
		#print(len(self.testdata_y))
		#temp10 = self.traindata_x[0]-self.testdata_x[198]
		#temp11 = numpy.abs(temp10)
		#print(numpy.max(temp11))		
		print(self.resultmat)



	def predict(self):
		j = 0
		temparr = self.resultmat[j,:,0:2]
		print(temparr)
		temparr2 = sorted(temparr, key=lambda x:x[1])
		print(temparr2[0])
		print(temparr2[1])
		print(temparr2[0][0])
		print(temparr2[1][1])
		print(self.k)
		

	#def LLoop(self):


