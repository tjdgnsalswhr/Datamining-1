import pandas
import imp
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


with open("cancer_train.csv", 'r') as csvfile:
	traindata = pandas.read_csv(csvfile,names=['label','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses'])

#print(traindata)

with open("cancer_test.csv", 'r') as csvfile2:
	testdata = pandas.read_csv(csvfile2,names=['label','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses'])

#print(testdata)

traindata_x = traindata.iloc[:,1:10].values
traindata_y = traindata.label.values

print(traindata_x)
#print(traindata_y)

knn5 = KNeighborsClassifier(n_neighbors=3) 
knn5.fit(traindata_x,traindata_y) 

testdata_x = testdata.iloc[:,1:10].values
testdata_y = testdata.label.values


knnpredict_y = knn5.predict(testdata_x)

print(accuracy_score(testdata_y,knnpredict_y))
print(knn5.score(testdata_x,testdata_y))
#----------------------------------------------------------------

decstree = DecisionTreeClassifier()
decstree.fit(traindata_x, traindata_y)
treepredict_y = decstree.predict(testdata_x)

print(accuracy_score(testdata_y,treepredict_y))


#----------------------------------------------------------------


svmachine = SVC(gamma=0.10)
svmachine.fit(traindata_x, traindata_y)
svmpredict_y = svmachine.predict(testdata_x)

print(accuracy_score(testdata_y, svmpredict_y))



print("--------------------KNN Cross Validation---------------------------------------\n\n\n\n")


i = range(1,30)
scorelist = []
for k in i:
	tempknn = KNeighborsClassifier(n_neighbors=k)
	#tempkfold = KFold(shuffle=True, random_state=0)
	score = cross_val_score(tempknn, traindata_x, traindata_y, cv=10)
	#print(score)
	avg = sum(score)/len(score)
	print("%d of KNeighborsClassifier's  mean of score is %f"  %(k,avg))


print("\n\n\n")
knn17 = KNeighborsClassifier(n_neighbors=17)
knn17.fit(traindata_x, traindata_y)
knn17pred_y = knn17.predict(testdata_x)
knn17confu = metrics.classification_report(testdata_y,knn17pred_y)
print(knn17confu)

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


