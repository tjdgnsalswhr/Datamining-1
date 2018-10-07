import numpy
import matplotlib


class Myclassifier():
	

	def __init__(self, k, traindata_x, traindata_y, testdata_x, testdata_y, vertical):
		self.k = k
		self.traindata_x = traindata_x
		self.traindata_y = traindata_y
		self_testdata_x = testdata_x
		self_testdata_y = testdata_y
		self.vertical = vertical
		self.resultmat = numpy.zeros((self.vertical,3))
	#Training..............

	def learning(self):
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


#def Get_Distance(traindata_x, traindata_y

#def Euclidian(exist_x,input_x):
	
	
	


#Get_Manhattan Distance-----------------------






#Get_Infinite L Distance-----------------------





#Predict--------------------
