import pandas
from problem2_knn import Myclassifier
import numpy

#Problem[2] 2-A

with open("digits_train.csv",'r') as csvfile:
	traindata = pandas.read_csv(csvfile, header = None)

with open("digits_test.csv",'r') as csvfile2:
	testdata = pandas.read_csv(csvfile2, header = None)

#Problem[2] 2-C

def fivefoldCV(data_x,data_y):
	
	print("\n\n----------** 5-Fold Cross Validation will be executed **--------------\n\n")
	scorelist = []
	sliced_x = []
	sliced_y = []
	for i in range(0,5):
		start = int((len(data_y))/5*i)
		final = int((len(data_y))/5*(i+1))
		sliced_x.append(data_x[start:final])
		sliced_y.append(data_y[start:final])
	x = numpy.array(sliced_x)
	y = numpy.array(sliced_y)	

	for i in range(0,5):
		temptrainx = []
		temptrainy = []
		temptestx = []
		temptesty = []
		for j in range(0,5):
			if i!=j:
				temptrainx.append(x[j])
				temptrainy.append(y[j])
			else:
				temptestx.append(x[j])
				temptesty.append(y[j])
	
		trainx = numpy.array(temptrainx[0])
		for z in range(1,4):
			temparr = numpy.array(temptrainx[z])
			trainx = numpy.vstack((trainx,temparr))

			
		trainy = (numpy.array(temptrainy)).reshape(-1)
		testx = (numpy.array(temptestx[0]))
		testy = (numpy.array(temptesty)).reshape(-1)


		tempknn = Myclassifier(5,trainx,trainy)
		tempknn.learning(testx,testy)

		predy = tempknn.predict()
		trecall, tprecision, tfmeasure, taccuracy = tempknn.score(testy,predy)
		scorelist.append(taccuracy)

	print("\n\n----------**5Fold result Accuracy list is below **---------------\n\n")
	print(scorelist)
	scoreavg = sum(scorelist)/len(scorelist)
	print("\n\nThis Classifier's average accuracy is : %f\n\n"%scoreavg)

	return scoreavg


#Main Function
	
traindata_x = traindata.iloc[:,1:785].values
traindata_y = (traindata.iloc[:,0:1].values).reshape(-1)

testdata_x = testdata.iloc[:,1:785].values
testdata_y = (testdata.iloc[:,0:1].values).reshape(-1)

myknn = Myclassifier(10, traindata_x, traindata_y)

myknn.learning(testdata_x, testdata_y)
predict_y = myknn.predict()

recall,precision,fmeasure,accuracy = myknn.score(testdata_y,predict_y)


result = fivefoldCV(traindata_x,traindata_y)




