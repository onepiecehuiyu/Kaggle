# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from datetime import datetime
import pandas, numpy
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

def address(data):
	it = [w for w in data.to_string().split() if w.isupper() and len(w) > 2]
	tr1 = it[0]
	tr2 = it[0]
	if len(it) != 1:
		tr2 = it[1]
	return tr1, tr2



def parse_time(x):
	DD = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
	time = DD.hour  # *60+DD.minute
	day = DD.day
	month = DD.month
	year = DD.year
	return time, day, month, year



# dataset = pandas.read_csv("D:\\code_python\\ML\\San Francisco Crime Classification\\train.csv")
# # dataset['tr1'], dataset['tr2'] = zip(*dataset[['Address']].apply(func=address, axis=1))
# # numpy.savetxt("AddressTostreet.csv", dataset[['tr1', 'tr2']], delimiter=',', fmt='%s')
# dataset = pandas.get_dummies(dataset['DayOfWeek']).join(dataset)
# dataset = pandas.get_dummies(dataset['PdDistrict']).join(dataset)
# # dataset = pandas.get_dummies(dataset['tr1']).join(dataset)
# # dataset = pandas.get_dummies(dataset['tr2']).join(dataset)
# dataset['time'], dataset['day'], dataset['month'], dataset['year'] = zip(*dataset['Dates'].apply(func=parse_time))
# x = preprocessing.MinMaxScaler().fit_transform(dataset['X'].values)
# dataset = pandas.DataFrame(data=x, columns=['xx']).join(dataset)
# y = preprocessing.MinMaxScaler().fit_transform(dataset['Y'].values)
# labelData = dataset['Category'].values
# dataset = pandas.DataFrame(data=y, columns=['yy']).join(dataset)
# dataset = pandas.get_dummies(dataset['time'], prefix='time').join(dataset)
# dataset = pandas.get_dummies(dataset['day'], prefix='day').join(dataset)
# dataset = pandas.get_dummies(dataset['month'], prefix='month').join(dataset)
# dataset = pandas.get_dummies(dataset['year'], prefix='year').join(dataset)
# dataset.drop(['X', 'Y', 'DayOfWeek', 'PdDistrict', 'Address',
#              'time', 'day', 'month', 'year','Dates',
#              'Resolution', 'Descript', 'Category'], inplace=True, axis=1)

testset = pandas.read_csv("D:\\code_python\\ML\\San Francisco Crime Classification\\test.csv")
# testset['tr1'], testset['tr2'] = zip(*testset[['Address']].apply(func=address, axis=1))
testset = pandas.get_dummies(testset['DayOfWeek']).join(testset)
testset = pandas.get_dummies(testset['PdDistrict']).join(testset)
# testset = pandas.get_dummies(testset['tr1']).join(testset)
# testset = pandas.get_dummies(testset['tr2']).join(testset)
testset['time'], testset['day'], testset['month'], testset['year'] = zip(*testset['Dates'].apply(func=parse_time))
testset = pandas.get_dummies(testset['time'], prefix='time').join(testset)
testset = pandas.get_dummies(testset['day'], prefix='day').join(testset)
testset = pandas.get_dummies(testset['month'], prefix='month').join(testset)
testset = pandas.get_dummies(testset['year'], prefix='year').join(testset)
x = preprocessing.MinMaxScaler().fit_transform(testset['X'].values)
testset = pandas.DataFrame(data=x, columns=['xx']).join(testset)
y = preprocessing.MinMaxScaler().fit_transform(testset['Y'].values)
testset = pandas.DataFrame(data=y, columns=['yy']).join(testset)
testset.drop(['X', 'Y', 'DayOfWeek', 'PdDistrict', 'Address',
             'time', 'day', 'month', 'year', 'Dates', 'Id'], inplace=True, axis=1)

# trainData = dataset.iloc[:, :].values
testData = testset.iloc[:, :].values

# numpy.savetxt("D:\\code_python\\ML\\San Francisco Crime Classification\\train2.csv", trainData,
#              delimiter=',', header=','.join(['%s' % num for num in dataset]), comments='', fmt='%s')
numpy.savetxt("D:\\code_python\\ML\\San Francisco Crime Classification\\test2.csv", testset,
             delimiter=',', header=','.join(['%s' % num for num in testset]), comments='', fmt='%s')
# Model
# LRModel = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial')
#
# #CV
# print(cross_validation.cross_val_score(LRModel, trainData, labelData, cv=5).mean())
