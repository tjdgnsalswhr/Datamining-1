import pandas
import imp
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


with open("cancer_train.csv", 'r') as csvfile:
	traindata = pandas.read_csv(csvfile,names=['label','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses'])


with open("cancer_test.csv", 'r') as csvfile2:
	testdata = pandas.read_csv(csvfile2,names=['label','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses'])


print("------------------**  Data is loaded **----------------------------\n\n")

print("-----------------** x of training data **-------------------------------\n\n")

traindata_x = traindata.iloc[:,1:10].values
print(traindata_x)

print("\n\n-------------------** y of training data **--------------------------\n\n")
traindata_y = traindata.label.values
print(traindata_y)

print("\n\n-------------------** x of test data **--------------------------------\n\n")
testdata_x = testdata.iloc[:,1:10].values
print(testdata_x)

print("\n\n-------------------** y of test data **----------------------------------\n\n")
testdata_y = testdata.label.values
print(testdata_y)


#problem 1-A

print("\n\n--------------------** Use the KNeighborsClassifier **-----------------------\n\n")

knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(traindata_x,traindata_y)
knnpredict_y = knn5.predict(testdata_x)

knntrainaccuracy = knn5.score(traindata_x,traindata_y)
print("The 5-NN Classifier's Train Accuracy is %f\n"%knntrainaccuracy)
knntestaccuracy = knn5.score(testdata_x, testdata_y)
print("The 5-NN Classifier's Test Accuracy is %f\n"%knntestaccuracy)

#problem 1-B

print("\n\n-------------------** Use the DicisionTreeClassifier **---------------------\n\n")

decstree = DecisionTreeClassifier()
decstree.fit(traindata_x, traindata_y)
treepredict_y = decstree.predict(testdata_x)

dcstrainaccuracy = decstree.score(traindata_x, traindata_y)
print("The Dicisiontree Classifier's Train Accuracy is %f\n"%dcstrainaccuracy)
dcstestaccuracy = decstree.score(testdata_x, testdata_y)
print("The Dicisiontree Classifier's Test Accuracy is %f\n"%dcstestaccuracy)



#problem 1-C

print("\n\n------------------** Use Support Vector Machine ** ---------------------\n\n")
svmachine = SVC(gamma=0.001, C=100.)
svmachine.fit(traindata_x, traindata_y)
svmpredict_y = svmachine.predict(testdata_x)

svmtrainaccuracy = svmachine.score(traindata_x, traindata_y)
print("The Support Vector Machine's Train Accuracy is %f\n"%svmtrainaccuracy)
svmtestaccuracy = svmachine.score(testdata_x, testdata_y)
print("The Support Vector Machine's Test Accuracy is %f\n"%svmtestaccuracy)



#problem 1-D-1
print("\n\n--------** Use Cross Validation for finding hyper parameter K **-----------\n\n")

scorelist = []
for k in range(3,31):
	tempknn = KNeighborsClassifier(n_neighbors=k)
	score = cross_val_score(tempknn, traindata_x, traindata_y, cv=5)
	mean = sum(score)/len(score)
	print("In 5Fold Cross Validation, the %dNN Classifier's mean of accuracy : %f"%(k,mean))

print("\n\nIn above result, when K is 6, the mean of accurcy is maximum value\n\n")

#problem 1-E-1

print("\n\n--------------------** 6-NN Classifier Analysis **--------------------\n\n")
knn6 = KNeighborsClassifier(n_neighbors=6)
knn6.fit(traindata_x, traindata_y)
knn6pred_y = knn6.predict(testdata_x)
analysis = metrics.classification_report(testdata_y,knn6pred_y)
confusion = confusion_matrix(testdata_y, knn6pred_y)

print("Confusion Matrix is : \n\n")
print(confusion)
print("\n\nAnalysis is : \n\n")
print(analysis)

#problem 1-D-2

print("\n\n-------** Use Cross Validation for finding hyper parameter Gamma **-----------\n\n")


for j in range(1,30):
	tempsvm = SVC(gamma=0.005, C=100.)
	score = cross_val_score(tempsvm, traindata_x, traindata_y, cv=5)
	mean = sum(score)/len(score)
	print("In 5Fold Cross Validation, the SVM to have %f Gamma's mean of accuracy : %f"%(j,mean))
	print("\n")

print("\n\n\n")
'''
svm?? = SVC(gamma=??)
svm??.fit(traindata_x, traindata_y)
svm??pred_y = svm??.predict(testdata_x)
svm??confu = metrics.classification_report(testdata_y,svm??pred_y)
'''




