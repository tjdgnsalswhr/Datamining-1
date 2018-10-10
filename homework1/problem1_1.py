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




print("\n\n--------** Use Cross Validation for finding hyper parameter K **-----------\n\n")


i = range(1,30)
scorelist = []
for k in i:
	tempknn = KNeighborsClassifier(n_neighbors=k)
	#tempkfold = KFold(shuffle=True, random_state=0)
	score = cross_val_score(tempknn, traindata_x, traindata_y, cv=10)
	print(score)
	avg = sum(score)/len(score)
	print("%d of KNeighborsClassifier's  mean of score is %f"  %(k,avg))


print("\n\n\n")
knn17 = KNeighborsClassifier(n_neighbors=17)
knn17.fit(traindata_x, traindata_y)
knn17pred_y = knn17.predict(testdata_x)
knn17confu = metrics.classification_report(testdata_y,knn17pred_y)
confusion = confusion_matrix(testdata_y, knn17pred_y)

print(knn17confu)
print(confusion)
"""

print("--------------------SVM Cross Validation---------------------------------------\n\n\n\n")


i = range(1,30)
for j in i:
        tempsvm = SVC(gamma=j)
        #tempkfold = KFold(shuffle=True, random_state=0)
        score = cross_val_score(tempsvm, traindata_x, traindata_y, cv=10)
        #print(score)
        avg = sum(score)/len(score)
        print("%f of gamma of SVM's mean of score is %f"  %(j,avg))


print("\n\n\n")
svm?? = SVC(gamma=??)
svm??.fit(traindata_x, traindata_y)
svm??pred_y = svm??.predict(testdata_x)
svm??confu = metrics.classification_report(testdata_y,svm??pred_y)


"""


