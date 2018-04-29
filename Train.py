import numpy as np
import csv
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm

data4 = []
data5 = []
data6 = []
data7 = []

with open('TrainDataFinal.csv') as f4:
    fid4 = csv.reader(f4)
    for row in fid4:
        data4.append(row[0:])
with open('TrainLabelFinal.csv') as f5:
    fid5 = csv.reader(f5)
    for row in fid5:
        data5.append(row[0:])
with open('TestDataFinal.csv') as f6:
    fid6 = csv.reader(f6)
    for row in fid6:
        data6.append(row[0:])
with open('Sample_submission.csv') as f7:
    fid7 = csv.reader(f7)
    for row in fid7:
        data7.append(row[0:])

data4.pop(0)
data5.pop(0)
data6.pop(0)
data7.pop(0)

TrainData = np.array(data4)
TrainLabel = np.array(data5)
TestData = np.array(data6)
Output = np.array(data7)
TrainData = TrainData.astype('int32')
TrainLabel = TrainLabel.astype('int32')


'''
train set: 0.8, test set: 0.2
'''
train, test, train_label, test_label = train_test_split(TrainData, TrainLabel, test_size = 0.2)

'''
KNN Classifier: nearest neighbors = 5
'''
knn = KNeighborsClassifier(n_neighbors=5)
train_label = train_label.ravel()
knn.fit(train, train_label)

'''
Random Forest
'''
rf = RandomForestClassifier()
train_label = train_label.ravel()
rf.fit(train, train_label)

'''
GBDT
'''
gbdt = GradientBoostingClassifier()
train_label = train_label.ravel()
gbdt.fit(train, train_label)

'''
Naive Bayes
'''
nb = GaussianNB()
train_label = train_label.ravel()
nb.fit(train, train_label)

'''
Adaboost
'''
ad = AdaBoostClassifier()
train_label = train_label.ravel()
ad.fit(train, train_label)

'''
SVM
'''
SVM = svm.SVC()
train_label = train_label.ravel()
SVM.fit(train, train_label)

'''
Print the accuracy of training and testing
'''
print('The training accuracy of KNN is ',knn.score(train,train_label))
print('The testing accuracy of KNN is ',knn.score(test,test_label),"\n")
print('The training accuracy of Random Forest is ',rf.score(train,train_label))
print('The testing accuracy of Random Forest is ',rf.score(test,test_label),"\n")
print('The training accuracy of GBDT is ',gbdt.score(train,train_label))
print('The testing accuracy of GBDT is ',gbdt.score(test,test_label),"\n")
print('The training accuracy of Naive Bayes is ',nb.score(train,train_label))
print('The testing accuracy of Naive Bayes is ',nb.score(test,test_label),"\n")
print('The training accuracy of Adaboost is ',ad.score(train,train_label))
print('The testing accuracy of Adaboost is ',ad.score(test,test_label),"\n")
print('The training accuracy of SVM is ',SVM.score(train,train_label))
print('The testing accuracy of SVM is ',SVM.score(test,test_label),"\n")

'''
Prediction
'''
Test_predict = gbdt.predict(TestData) # predict the test data
PredictFrame = pd.DataFrame(Test_predict,columns=['prediction'])

le = preprocessing.LabelEncoder()
le.fit(Output[:,0]) # Map the user_id into numeric id
Output[:,1] = le.transform(Output[:,0])
OutputFrame = pd.DataFrame(Output,columns=['user_id','numeric_id'])
OutputFrame['numeric_id'] = OutputFrame['numeric_id'].astype('int') # transform type 'string' to 'int'
OutputFrame = OutputFrame.sort_values(by=['numeric_id'])
OutputFrame = OutputFrame.join(PredictFrame)
OutputFrame.pop('numeric_id')
OutputFrame.to_csv('prediction.csv',index=False)


