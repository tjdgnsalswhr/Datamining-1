import pandas
from problem2_knn import Myclassifier

with open("digits_train.csv",'r') as csvfile:
	traindata = pandas.read_csv(csvfile, header = None)

with open("digits_test.csv",'r') as csvfile2:
	testdata = pandas.read_csv(csvfile2, header = None)

traindata_x = traindata.iloc[:,1:785].values
traindata_y = (traindata.iloc[:,0:1].values).reshape(-1)
#print(traindata_y)
testdata_x = testdata.iloc[:,1:785].values
testdata_y = (testdata.iloc[:,0:1].values).reshape(-1)


myknn = Myclassifier(17, traindata_x, traindata_y)

#print(myknn.k)

myknn.learning(testdata_x, testdata_y)
predict_y = myknn.predict()

#print(traindata_y)
#print(predict_y)

myknn.score(testdata_y,predict_y)






#yknn.predict()
