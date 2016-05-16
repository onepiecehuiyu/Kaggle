__author__ = 'Whiker'
__mtime__ = '2016/5/15'

import pandas, numpy
from sklearn.ensemble import RandomForestClassifier

dataset = pandas.read_csv("train.csv")
testset = pandas.read_csv("test.csv")
dataset = dataset.fillna(0)
traindata = dataset.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
labeldata = dataset.Survived.values
traindata['Gender'] = traindata['Sex'].map({'male': 0, 'female': 1}).astype(int)
traindata.loc[traindata['Pclass'] == 1, 'Pclass'] = 0
traindata.loc[traindata['Pclass'] == 2, 'Pclass'] = 1
traindata.loc[traindata['Pclass'] == 3, 'Pclass'] = 2
traindata = traindata.drop(['Sex'], axis=1).values

testset = testset.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
testset = testset.fillna(0)
testset['Gender'] = testset['Sex'].map({'male': 0, 'female': 1}).astype(int)
testset.loc[testset['Pclass'] == 1, 'Pclass'] = 0
testset.loc[testset['Pclass'] == 2, 'Pclass'] = 1
testset.loc[testset['Pclass'] == 3, 'Pclass'] = 2
testData = testset.drop(['Sex'], axis=1).values

RFModel = RandomForestClassifier(n_estimators=100)
RFModel.fit(traindata, labeldata)
preds = RFModel.predict(testData)
numpy.savetxt('submission_rf.csv', numpy.c_[range(892, 891+len(testData) + 1), preds], delimiter=',',
			  header='PassengerId,Survived', comments='', fmt='%d')
