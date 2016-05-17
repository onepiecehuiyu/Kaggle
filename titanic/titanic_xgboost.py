__author__ = 'Whiker'
__mtime__ = '2016/5/17'

import pandas, xgboost, numpy

dataset = pandas.read_csv("train.csv")
testset = pandas.read_csv("test.csv")

labelData = dataset.Survived.values
trainData = dataset[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
testData = testset[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Pclass
trainData = pandas.get_dummies(trainData['Pclass'], prefix='Pclass').join(trainData)
testData = pandas.get_dummies(testData['Pclass'], prefix='Pclass').join(testData)

# Sex
trainData = pandas.get_dummies(trainData['Sex']).join(trainData)
testData = pandas.get_dummies(testData['Sex']).join(testData)

# Age
ageMean = trainData['Age'].mean()
trainData.Age[trainData['Age'].isnull()] = ageMean
ageMean = testData['Age'].mean()
testData.Age[testData.Age.isnull()] = ageMean

# SibSp Parch
trainData['Family'] = trainData['SibSp'] + trainData['Parch']
testData['Family'] = testData['SibSp'] + testData['Parch']

# Fare
fareMedian = testData['Fare'].median()
testData.Fare[testData.Fare.isnull()] = fareMedian

# Embarked
trainData['Embarked'] = trainData['Embarked'].fillna('S')
testData['Embarked'] = testData['Embarked'].fillna('S')

# delete feature
trainData.drop(['Pclass', 'Embarked', 'SibSp', 'Parch', 'Sex'], inplace=True, axis=1)
testData.drop(['Pclass', 'Embarked', 'SibSp', 'Parch', 'Sex'], inplace=True, axis=1)

trainData = trainData.values
testData = testData.values

offset = 600

xgModel = xgboost.XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=500)
xgModel.fit(trainData, labelData,
			eval_set=[(trainData[:offset, :], labelData[:offset]),
					  (trainData[offset:, :], labelData[offset:])], eval_metric='logloss', verbose=True)

preds = xgModel.predict(testData)

numpy.savetxt('submission_xgboost.csv', numpy.c_[range(892, 891 + len(testset) + 1), preds], delimiter=',',
			  header='PassengerId,Survived', comments='', fmt='%d')
