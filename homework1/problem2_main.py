import pandas
from problem2_knn import Myclassifier
import numpy


with open("digits_train.csv",'r') as csvfile:
	traindata = pandas.read_csv(csvfile, header = None)

with open("digits_test.csv",'r') as csvfile2:
	testdata = pandas.read_csv(csvfile2, header = None)


def fivefoldCV(data_x,data_y):
	print("This is in fivefoldCV")
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
		print(trainx)
		print(len(trainx))
			
		trainy = (numpy.array(temptrainy)).reshape(-1)
		testx = (numpy.array(temptestx[0]))
		testy = (numpy.array(temptesty)).reshape(-1)
		print(testx)
		print(trainy)
		print(testy)

		tempknn = Myclassifier(5,trainx,trainy)
		tempknn.learning(testx,testy)

		predy = tempknn.predict()
		trecall, tprecision, tfmeasure, taccuracy = tempknn.score(testy,predy)
		scorelist.append(taccuracy)


	print(scorelist)
	scoreavg = sum(scorelist)/len(scorelist)
	print(scoreavg)
	return scoreavg

	
traindata_x = traindata.iloc[:,1:785].values
traindata_y = (traindata.iloc[:,0:1].values).reshape(-1)
#print(traindata_x)
testdata_x = testdata.iloc[:,1:785].values
testdata_y = (testdata.iloc[:,0:1].values).reshape(-1)

print(traindata_x)
print(traindata_y)
print(testdata_x)
print(testdata_y)
myknn = Myclassifier(5, traindata_x, traindata_y)

#print(myknn.k)

myknn.learning(testdata_x, testdata_y)
predict_y = myknn.predict()

#print(traindata_y)
#print(predict_y)

recall,precision,fmeasure,accuracy = myknn.score(testdata_y,predict_y)

print("recall is %f" %recall)
print("precision is %f" %precision)
print("fmeasure is %f" %fmeasure)
print("accuracy is %f" %accuracy)


result = fivefoldCV(traindata_x,traindata_y)

print("The Fold result is %f" %result)





