__author__ = 'Whiker'
__mtime__ = '2016/5/18'

from datetime import datetime
import csv

import pandas
import numpy
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression


def parse_time(x):
    DD = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    time = DD.hour  # *60+DD.minute
    day = DD.day
    month = DD.month
    year = DD.year
    return time, day, month, year


dataset = pandas.read_csv("data/train.csv")
testset = pandas.read_csv("data/test.csv")
labelData = dataset['Category'].values
myId = testset['Id'].values

# DayOfWeek PdDistrict
dataset = pandas.get_dummies(dataset['DayOfWeek']).join(dataset)
dataset = pandas.get_dummies(dataset['PdDistrict']).join(dataset)

# Dates
dataset['time'], dataset['day'], dataset['month'], dataset['year'] = zip(*dataset['Dates'].apply(func=parse_time))
dataset = pandas.get_dummies(dataset['time'], prefix='time').join(dataset)
dataset = pandas.get_dummies(dataset['day'], prefix='day').join(dataset)
dataset = pandas.get_dummies(dataset['month'], prefix='month').join(dataset)
dataset = pandas.get_dummies(dataset['year'], prefix='year').join(dataset)

# X Y
x = preprocessing.MinMaxScaler().fit_transform(dataset['X'].values)
dataset = pandas.DataFrame(data=x, columns=['xx']).join(dataset)
y = preprocessing.MinMaxScaler().fit_transform(dataset['Y'].values)
dataset = pandas.DataFrame(data=y, columns=['yy']).join(dataset)

dataset.drop(['DayOfWeek', 'PdDistrict', 'Address', 'X', 'Y',
              'Dates', 'Resolution', 'Descript', 'Category'], inplace=True, axis=1)


# DayOfWeek PdDistrict
testset = pandas.get_dummies(testset['DayOfWeek']).join(testset)
testset = pandas.get_dummies(testset['PdDistrict']).join(testset)

# Dates
testset['time'], testset['day'], testset['month'], testset['year'] = zip(*testset['Dates'].apply(func=parse_time))
testset = pandas.get_dummies(testset['time'], prefix='time').join(testset)
testset = pandas.get_dummies(testset['day'], prefix='day').join(testset)
testset = pandas.get_dummies(testset['month'], prefix='month').join(testset)
testset = pandas.get_dummies(testset['year'], prefix='year').join(testset)

# X Y
x = preprocessing.MinMaxScaler().fit_transform(testset['X'].values)
testset = pandas.DataFrame(data=x, columns=['xx']).join(testset)
y = preprocessing.MinMaxScaler().fit_transform(testset['Y'].values)
testset = pandas.DataFrame(data=y, columns=['yy']).join(testset)

testset.drop(['DayOfWeek', 'PdDistrict', 'Address', 'X', 'Y',
              'Dates', 'Id'], inplace=True, axis=1)

trainData = dataset.iloc[:, :].values
testData = testset.iloc[:, :].values

LRModel = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', random_state=222, C=0.01)
LRModel.fit(dataset.iloc[:, :].values, labelData)
# print(LRModel.score(dataset.iloc[:, :].values, labelData))

preds = LRModel.predict_proba(testData)

cList = LRModel.classes_.tolist()
cList.insert(0, 'Id')
preds = numpy.column_stack((myId, preds)).tolist()
preds = [[int(i[0])] + i[1:] for i in preds]
csvfile = 'result/lr_sub_proba.csv'
with open(csvfile, 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(cList)
    writer.writerows(preds)
