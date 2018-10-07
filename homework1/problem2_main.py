import pandas
from problem2_knn import Myclassifier

with open("digits_train.csv",'r') as csvfile:
	traindata = pandas.read_csv(csvfile, header = None)

with open("digits_test.csv",'r') as csvfile2:
	testdata = pandas.read_csv(csvfile2, header = None)

traindata_x = traindata.iloc[:,1:785].values
traindata_y = traindata.iloc[:,0:1].values

testdata_x = testdata.iloc[:,1:785].values
testdata_y = testdata.iloc[:,0:1].values


myknn = Myclassifier(3, traindata_x, traindata_y, testdata_x, testdata_y, len(traindata_y))

print(myknn.k)

myknn.learning()
