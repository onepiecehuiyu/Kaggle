import csv
from math import log, exp

import numpy

__author__ = 'Whiker'
__mtime__ = '2016/5/26'

from datetime import datetime

import pandas
import xgboost


def parse_time(data):
	date = datetime.strptime(data, "%Y-%m-%d %H:%M:%S")
	hour = date.hour
	day = date.day
	month = date.month
	dow = date.weekday()
	year = date.year
	return year, month, day, hour, dow


def mylog(data):
	data = int(data) + 1
	return log(data)


dataset = pandas.read_csv("input/train.csv")
testset = pandas.read_csv("input/test.csv")

labelData = dataset['casual'].apply(func=mylog)
labelData2 = dataset['registered'].apply(func=mylog)
myIndex = testset['datetime']

feature1 = ['atemp', 'temp', 'hour', 'humidity', 'windspeed',
			'month', 'dow', 'workingday', 'holiday', 'year', 'weather']

# datetime
dataset['year'], dataset['month'], dataset['day'], dataset['hour'], dataset['dow'] = zip(
	*dataset['datetime'].apply(func=parse_time))
testset['year'], testset['month'], testset['day'], testset['hour'], testset['dow'] = zip(
	*testset['datetime'].apply(func=parse_time))


trainData = dataset[feature1].iloc[:, :].values
testData = testset[feature1].iloc[:, :].values

offset = 6000
xgtrain = xgboost.DMatrix(trainData[:offset, :], label=labelData[:offset])
xgeval = xgboost.DMatrix(trainData[offset:, :], label=labelData[offset:])
xgtest = xgboost.DMatrix(testData)

watchlist = [(xgtrain, 'train'), (xgeval, 'val')]
params = {"max_depth": 6, "tree_num": 1000, "silent": 1, "shrinkage": 0.1}

xgModel = xgboost.train(list(params.items()), xgtrain, 450, watchlist, early_stopping_rounds=100)
# preds = numpy.column_stack((myIndex, xgModel.predict(xgtest, ntree_limit=xgModel.best_iteration))).tolist()
preds = xgModel.predict(xgtest, ntree_limit=xgModel.best_iteration).tolist()
preds = [exp(i) - 1 for i in preds]


# registered ================================================
feature2 = ['hour', 'humidity', 'atemp', 'temp', 'windspeed',
			'month', 'dow', 'workingday', 'holiday', 'year', 'weather']
trainData = dataset[list(feature2)].iloc[:, :].values
testData = testset[list(feature2)].iloc[:, :].values

xgtrain = xgboost.DMatrix(trainData[:offset, :], label=labelData2[:offset])
xgeval = xgboost.DMatrix(trainData[offset:, :], label=labelData2[offset:])
xgtest = xgboost.DMatrix(testData)
watchlist = [(xgtrain, 'train'), (xgeval, 'val')]

params = {"max_depth": 6, "tree_num": 1000, "silent": 1, "shrinkage": 0.1}

xgModel = xgboost.train(list(params.items()), xgtrain, 450, watchlist, early_stopping_rounds=100)
preds2 = xgModel.predict(xgtest, ntree_limit=xgModel.best_iteration).tolist()
preds2 = [exp(i) - 1 for i in preds2]


# ==================================================================

preds = numpy.column_stack((myIndex, map(lambda x, y: x + y, preds, preds2))).tolist()

with open("result/sub_xgb_linear.csv", "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	writer.writerow(["datetime", "count"])
	writer.writerows(preds)
