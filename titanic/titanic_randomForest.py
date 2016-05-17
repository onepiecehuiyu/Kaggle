__author__ = 'Whiker'
__mtime__ = '2016/5/15'

import pandas, numpy
from sklearn.ensemble import RandomForestClassifier

dataset = pandas.read_csv("train.csv")
testset = pandas.read_csv("test.csv")
traindata = dataset.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
labeldata = dataset.Survived.values

# SibSp & Parch change to Family, Family > 1 = 1
traindata['Family'] = traindata['SibSp'] + traindata['Parch']
traindata.Family[traindata['Family'] > 0] = 1

# Age nan change to random(mean-std, mean+std)
ageMean = traindata['Age'].mean()
ageStd = traindata['Age'].std()
count_nan_age = traindata['Age'].isnull().sum()
rand_age = numpy.random.randint(ageMean - ageStd, ageMean + ageStd, count_nan_age)
traindata.Age[traindata.Age.isnull()] = rand_age
traindata['Age'] = traindata['Age'].astype(int)

# Sex age<=16 change to child, delete male
traindata.Sex[traindata.Age <= 16] = "child"
traindata = pandas.get_dummies(traindata['Sex']).join(traindata)

# Pclass delete classNum == 3
traindata = pandas.get_dummies(traindata['Pclass'], prefix='Pclass').join(traindata)

# Fare nan change to median
median = traindata['Fare'].median()
traindata['Fare'][traindata['Fare'].isnull()] = median
traindata['Fare'] = traindata['Fare'].astype(int)

# Embarked delete S
traindata['Embarked'] = traindata['Embarked'].fillna('S')
traindata = pandas.get_dummies(traindata['Embarked']).join(traindata)
traindata.drop(['Sex', 'Pclass', 'male', 'Pclass_3', 'Parch', 'SibSp', 'S', 'Embarked'], axis=1, inplace=True)

testset = testset.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
testset['Family'] = testset['SibSp'] + testset['Parch']
testset.Family[testset['Family'] > 0] = 1
ageMean = testset['Age'].mean()
ageStd = testset['Age'].std()
count_nan_age = testset['Age'].isnull().sum()
rand_age = numpy.random.randint(ageMean - ageStd, ageMean + ageStd, count_nan_age)
testset.Age[testset.Age.isnull()] = rand_age
testset['Age'] = testset['Age'].astype(int)
testset.Sex[testset.Age <= 16] = 'child'
testset = pandas.get_dummies(testset['Sex']).join(testset)
testset = pandas.get_dummies(testset['Pclass'], prefix='Pclass').join(testset)
median = testset['Fare'].median()
testset['Fare'][testset['Fare'].isnull()] = median
testset['Fare'] = testset['Fare'].astype(int)
testset['Embarked'] = testset['Embarked'].fillna('S')
testset = pandas.get_dummies(testset['Embarked']).join(testset)
testset.drop(['Sex', 'Pclass', 'male', 'Pclass_3', 'SibSp', 'Parch', 'S', 'Embarked'], axis=1, inplace=True)

RFModel = RandomForestClassifier(random_state=1,
								 n_estimators=150,
								 min_samples_split=4,
								 min_samples_leaf=2)
RFModel.fit(traindata, labeldata)
preds = RFModel.predict(testset)
numpy.savetxt('submission_rf.csv', numpy.c_[range(892, 891 + len(testset) + 1), preds], delimiter=',',
			  header='PassengerId,Survived', comments='', fmt='%d')
