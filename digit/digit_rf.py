# coding:utf-8
__author__ = 'WHP'
__mtime__ = '2016/5/13'
__name__ = ''
import pandas, numpy
from sklearn.ensemble import RandomForestClassifier

dataset = pandas.read_csv("C:\\Users\\Whiker\\Desktop\\kaggle\\digit recognizer\\data\\train.csv")
trainData = dataset.iloc[:, 1:].values
labelData = dataset.label.values
testDataset = pandas.read_csv("C:\\Users\\Whiker\\Desktop\\kaggle\\digit recognizer\\data\\test.csv")
testData = testDataset.iloc[:, :].values

rfModel = RandomForestClassifier(n_estimators=100)
rfModel.fit(trainData, labelData)
preds = rfModel.predict(testData)
numpy.savetxt('submission_rf_MultiSoftmax.csv', numpy.c_[range(1, len(testData) + 1), preds], delimiter=',',
              header='ImageId,Label', comments='', fmt='%d')

# import scipy
# import numpy as np
# import operator
# import csv
# from sklearn.ensemble import RandomForestClassifier
#
#
# def read_data(f, header=True, test=False, rows=0):
# 	data = []
# 	labels = []
#
# 	csv_reader = csv.reader(open(f, "r"), delimiter=",")
# 	index = 0
# 	for row in csv_reader:
# 		index = index + 1
# 		if rows > 0 & index > rows:
# 			break
# 		if header and index == 1:
# 			continue
#
# 		if not test:
# 			labels.append(int(row[0]))
# 			row = row[1:]
#
# 		data.append(np.array(np.int64(row)))
# 	return (data, labels)
#
#
# def predictRF(train, labels, test):
# 	print 'predicting...'
# 	rf = RandomForestClassifier(n_estimators=200, n_jobs=2)
# 	rf.fit(train, labels)
# 	print 'done fitting...'
# 	rf_predictions = rf.predict(test)
# 	rf_probs = rf.predict_proba(test)
# 	rf_BestProbs = rf_probs.max(axis=1)
# 	print('done with random forest.  Save text!')
# 	return rf_predictions, rf_BestProbs
#
#
# class PredScore:
# 	def __init__(self, prediction, score):
# 		self.prediction = prediction
# 		self.score = score
#
# 	prediction = -1
# 	score = 0
#
#
# x = []
# y = []
#
# fd = open('C:\\Users\\Whiker\\Desktop\\kaggle\\digit recognizer\\data\\train.csv', 'r')
# lines = fd.readlines()
# fd.close()
#
# for line in lines[1:]:
# 	data = line.split(',')
# 	x.append([int(i.strip()) for i in data[1:]])
# 	y.append(data[0])
# X = np.array(x)
# Y = np.array(y)
# fd = open('C:\\Users\\Whiker\\Desktop\\kaggle\\digit recognizer\\data\\test.csv', 'r')
# rfPredictions, rfScore = predictRF(X, Y, fd)
# retArray = []
# index = 0
# for rf in rfScore:
# 	rfPredScore = PredScore(rfPredictions[index], rfScore[index])
#
# 	options = []
# 	options.append(rfPredScore)
#
# 	maxObj = max(options, key=operator.attrgetter('score'))
# 	retArray.append(maxObj.prediction)
# 	index = index + 1
# np.savetxt('submission.csv', retArray, delimiter=',', fmt='%i')
