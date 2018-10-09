import numpy
import matplotlib
import operator

class Myclassifier():
	

	def __init__(self, k, traindata_x, traindata_y):
		self.k = k
		self.traindata_x = traindata_x
		self.traindata_y = traindata_y
		#print(self.traindata_y)
		#self_testdata_x = testdata_x
		#self_testdata_y = testdata_y
		#self.vertical = vertical
		#self.resultmat = numpy.full((len(self.testdata_y),self.vertical,3),0)
	#Training..............

	def learning(self, testdata_x,testdata_y):
		self.testdata_x = testdata_x
		self.testdata_y = testdata_y
		#print("This is in problem2-knn")
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

		self.resultmat = numpy.full((len(self.testdata_y),len(self.traindata_y),5),0.0)
		#self.resultmat[0][1][1] = 123
		#print(self.resultmat)
		for j in range(0,len(self.testdata_y)):
			for i in range(0,len(self.traindata_y)):
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
		#j = 2
		#temparr = self.resultmat[j,:,0:2]
		#print(temparr)
		#temparr2 = sorted(temparr, key=lambda x:x[1])
		#print(temparr2[0])
		#print(temparr2[1])
		#print(temparr2[2])
		#print(temparr2[3])
		#print(temparr2[4])
		#print(temparr2[5])
		#print(temparr2[6])
		pred_y = []
		for i in range(0,len(self.testdata_y)):
			temparr = self.resultmat[i,:,0:2]
			temparr2 = sorted(temparr, key=lambda x:x[1])
			
			labeltable = numpy.full((10),0)
			for j in range(0,self.k):
				#labeltable = numpy.full((10),0)
				index = int(temparr2[j][0])
				#print(index)
				labeltable[index] = labeltable[index]+1
			#print(labeltable)
			label = numpy.argmax(labeltable)
			#print(label)
			pred_y.append(label)
			#print(pred_y)	
					
		finpred_y = numpy.array(pred_y)	
		return finpred_y
		
		

	def score(self, actualdata, predictdata):
		print("This is in score function")
		confusion = numpy.full((10,10),0)
		self.actualdata = actualdata
		self.predictdata = predictdata
		print(self.actualdata)
		print(self.predictdata)
		for i in range(0,10):
			actualindex = (numpy.array(numpy.where(self.actualdata==i))).reshape(-1)
			print(actualindex)
			for j in actualindex:
				print(predictdata[j])
				if predictdata[j]==i:
					confusion[i][i] = confusion[i][i]+1
				else:
					confusion[i][predictdata[j]] = confusion[i][predictdata[j]]+1
		
		print(confusion)

		#--------------calculate recall---------------------------
		print("Calculate Recall")	
		tempsum = 0.0
		tempsum2 = 0.0
		for i in range(0,10):
			for j in range(0,10):
				tempsum = tempsum + confusion[i][j]
			print(tempsum)
			temprecall = confusion[i][i]/tempsum
			tempsum2 = tempsum2+temprecall
			tempsum = 0.0
		recall = tempsum2/10


		#-------------calculate precision------------------------

		print("Calculate Precision")
		tempsum = 0.0
		tempsum2 = 0.0
		for i in range(0,10):
			for j in range(0,10):
				tempsum = tempsum + confusion[j][i]
			print(tempsum)
			tempprecs = confusion[i][i]/tempsum
			tempsum2 = tempsum2+tempprecs
			tempsum=0.0
		precision = tempsum2/10

		#------------calculate F-1 score-----------------------

		print("Calculate F-1 score")
		fmeasure = 2*(precision*recall)/(precision+recall)

		#------------calculate Accuracy ----------------------
		print("Calculate Accuracy")
		tempsum = 0.0
		tempsum2 = 0.0
		for i in range(0,10):
			tempsum = tempsum + confusion[i][i]
		print(tempsum)
		tempsum2 = numpy.sum(confusion)
		accuracy = tempsum / tempsum2
		print(accuracy)
					
