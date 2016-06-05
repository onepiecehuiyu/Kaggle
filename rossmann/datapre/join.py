# -*- coding: UTF-8 -*-
import pandas

__author__ = 'Whiker'
__mtime__ = '2016/6/2'

dfs = pandas.read_csv("../input/store2.csv")
dtr = pandas.read_csv("../input/train2.csv")

df = pandas.merge(dfs, dtr, on='Store')
df.to_csv("../input/train2_.csv", index=False)

dte = pandas.read_csv("../input/test2.csv")
df = pandas.merge(dfs, dte, on='Store')
df.to_csv("../input/test2_.csv", index=False)
