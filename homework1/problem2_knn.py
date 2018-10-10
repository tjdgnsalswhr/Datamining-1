import numpy
import matplotlib
import operator


#problem[2]
class Myclassifier(): 
	
	#problem[2] 1-A
	def __init__(self, k, traindata_x, traindata_y):
		
		print("-----------** New Classifier is constructed **---------------\n\n")
		self.k = k
		self.traindata_x = traindata_x
		self.traindata_y = traindata_y


	def learning(self, testdata_x,testdata_y):

		print("\n\n-------------------** Training **------------------------\n\n")
		self.testdata_x = testdata_x
		self.testdata_y = testdata_y
		self.resultmat = numpy.full((len(self.testdata_y),len(self.traindata_y),5),0.0)
		for j in range(0,len(self.testdata_y)):
			for i in range(0,len(self.traindata_y)):
				self.resultmat[j][i][0] = self.traindata_y[i]
		print("\n\n This Matrix will have information of distance and label \n\n")
		print(self.resultmat)
		
		self.Getdistance()

	#problem[2] 1-B
	def Getdistance(self):
		
		print("\n\n --------------** Calculating distance **----------------\n\n")
		self.GetEuclidian()
		self.GetManhattan()
		self.GetInfiniteL()
		print("\n\n ------------** All calculating distance are complete! **------\n\n")
	
	def GetEuclidian(self):
		print("\n\n ------** Euclidian distance will be filled in matrix **-------\n\n")
		for i in range(0,len(self.testdata_y)):
			for j in range(0,len(self.traindata_y)):
				temp = self.traindata_x[j]-self.testdata_x[i]
				temp2 = temp**2
				tempsum = numpy.sum(temp2)
				tempresult = numpy.sqrt(tempsum)
				self.resultmat[i][j][1] = tempresult

		print(self.resultmat)

	def GetManhattan(self):
		print("\n\n -----** Manhattan distance will be filled in matrix **--------\n\n")
		for i in range(0,len(self.testdata_y)):
                        for j in range(0,len(self.traindata_y)):
                                temp = self.traindata_x[j]-self.testdata_x[i]
                                temp2 = numpy.abs(temp)
                                tempsum = numpy.sum(temp2)
                                self.resultmat[i][j][2] = tempsum
		
		print(self.resultmat)


	def GetInfiniteL(self):
		print("\n\n -----** Infinite L distance will be filled in matrix **-------\n\n")
		for i in range(0,len(self.testdata_y)):
			for j in range(0,len(self.traindata_y)):
				temp = self.traindata_x[j]-self.testdata_x[i]
				temp2 = numpy.abs(temp)
				tempresult = numpy.max(temp2)
				self.resultmat[i][j][3] = tempresult
	
		print(self.resultmat)


	#problem[2] 1-C

	def predict(self):
		
		print("\n\n--------------** label will be predicted **----------------\n\n")
		pred_y = []
		for i in range(0,len(self.testdata_y)):
			temparr = self.resultmat[i,:,0:2]
			temparr2 = sorted(temparr, key=lambda x:x[1])
			
			labeltable = numpy.full((10),0)
			for j in range(0,self.k):
				index = int(temparr2[j][0])
				labeltable[index] = labeltable[index]+1
			label = numpy.argmax(labeltable)
			pred_y.append(label)
					
		finpred_y = numpy.array(pred_y)

		print("\n\n--------------** predict data is below **-----------------\n\n")
		print(finpred_y)
		return finpred_y
		

	#problem[2] 2-B		

	def score(self, actualdata, predictdata):
		print("\n\n ---------------** Score will be caculated **--------------\n\n")
		confusion = numpy.full((10,10),0)
		self.actualdata = actualdata
		self.predictdata = predictdata
		for i in range(0,10):
			actualindex = (numpy.array(numpy.where(self.actualdata==i))).reshape(-1)
			for j in actualindex:
				if predictdata[j]==i:
					confusion[i][i] = confusion[i][i]+1
				else:
					confusion[i][predictdata[j]] = confusion[i][predictdata[j]]+1
		
		
		print("\n\n-------------** Confusion table is below **----------------\n\n")
		print(confusion)


		print("\n\n------------** Recall will be calculated **----------------\n\n")
		tempsum = 0.0
		tempsum2 = 0.0
		for i in range(0,10):
			for j in range(0,10):
				tempsum = tempsum + confusion[i][j]
			temprecall = confusion[i][i]/tempsum
			tempsum2 = tempsum2+temprecall
			tempsum = 0.0

		recall = tempsum2/10
		

		print("\n\n------------** Precision will be calculated **-------------\n\n")

		tempsum = 0.0
		tempsum2 = 0.0
		for i in range(0,10):
			for j in range(0,10):
				tempsum = tempsum + confusion[j][i]
			tempprecs = confusion[i][i]/tempsum
			tempsum2 = tempsum2+tempprecs
			tempsum=0.0
		precision = tempsum2/10

		print("\n\n-----------** F-Measure will be calculated **--------------\n\n")

		fmeasure = 2*(precision*recall)/(precision+recall)

		print("\n\n------------** Accuracy will be calculated **--------------\n\n")
		tempsum = 0.0
		tempsum2 = 0.0
		for i in range(0,10):
			tempsum = tempsum + confusion[i][i]
	
		tempsum2 = numpy.sum(confusion)
		accuracy = tempsum / tempsum2

		print("\n\n-----------** Score Result **--------------------\n\n")
		print("Recall is : %f" %recall)
		print("Precision is : %f"%precision)
		print("F-measure is : %f"%fmeasure)
		print("Accuracy is : %f"%accuracy)

		return recall,precision,fmeasure,accuracy		
