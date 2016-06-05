# -*- coding: UTF-8 -*-
import csv
from math import log, exp

import numpy
import pandas
import xgboost

__author__ = 'Whiker'
__mtime__ = '2016/6/2'


def sale(data):
	data = int(data) + 1
	return log(data)


dataset = pandas.read_csv("input/train2_.csv")
testset = pandas.read_csv("input/test2_.csv")

dataset['Sale'] = dataset['Sales'].apply(sale)

labelData = dataset['Sale'].values
myId = testset['Id'].values

testset.drop(['Id'], inplace=True, axis=1)
testData = testset.iloc[:, :].values
dataset.drop(['Sales', 'Sale'], inplace=True, axis=1)
dataData = dataset.iloc[:, :].values

offset = 800000
xgtrain = xgboost.DMatrix(dataData[:offset, :], labelData[:offset])
xgeval = xgboost.DMatrix(dataData[offset:, :], labelData[offset:])
xgtest = xgboost.DMatrix(testData)
watchlist = [(xgtrain, 'train'), (xgeval, 'val')]
params = {"max_depth": 5, "tree_num": 100, "silent": 1, "shrinkage": 0.1, "eval_metric": "rmse", "seed": 2016}
xgModel = xgboost.train(list(params.items()), xgtrain, 450, watchlist, early_stopping_rounds=100)
preds = xgModel.predict(xgtest, ntree_limit=xgModel.best_iteration).tolist()
preds = numpy.column_stack((myId, xgModel.predict(xgtest, ntree_limit=xgModel.best_iteration))).tolist()
preds = [[int(i[0])] + [exp(float(i[1])) - 1] for i in preds]

with open("result/sub_xgb_linear.csv", "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	writer.writerow(["Id", "Sales"])
	writer.writerows(preds)
