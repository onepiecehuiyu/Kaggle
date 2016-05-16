# coding:utf-8
import numpy

__author__ = 'WHP'
__mtime__ = '2016/5/12'
__name__ = ''

import xgboost
import pandas
import time

now = time.time()

dataset = pandas.read_csv("C:\\Users\\Whiker\\Desktop\\kaggle\\digit recognizer\\data\\train.csv")
trainData = dataset.iloc[1:, 1:].values
labelData = dataset.iloc[1:, :1].values
testData = pandas.read_csv("C:\\Users\\Whiker\\Desktop\\kaggle\\digit recognizer\\data\\test.csv")
test = testData.iloc[:, :].values

for i in range(len(trainData)-1):
    for j in range(len(trainData[0])-1):
        if trainData[i][j] > 0:
            trainData[i][j] = 1
for i in range(len(test)-1):
    for j in range(len(test[0])-1):
        if test[i][j] > 0:
            test[i][j] = 1

param = {"booster": "gbtree", "max_depth": 12, "eta": 0.03, "seed": 710, "objective": "multi:softmax", "num_class": 10,
         "gamma": 0.03}

offset = 35000
num_rounds = 500

xgtest = xgboost.DMatrix(test)
xgtrain = xgboost.DMatrix(trainData[:offset, :], label=labelData[:offset])
xgeval = xgboost.DMatrix(trainData[offset:, :], label=labelData[offset:])

watchlist = [(xgtrain, 'train'), (xgeval, 'val')]
model = xgboost.train(list(param.items()), xgtrain, num_rounds, watchlist, early_stopping_rounds=100)
preds = model.predict(xgtest, ntree_limit=model.best_iteration)
numpy.savetxt('submission_xgb_MultiSoftmax.csv', numpy.c_[range(1, len(testData) + 1), preds], delimiter=',',
              header='ImageId,Label', comments='', fmt='%d')

print("cost time:", time.time() - now)
