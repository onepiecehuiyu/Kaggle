# -*- coding: UTF-8 -*-
from datetime import datetime

import pandas

__author__ = 'Whiker'
__mtime__ = '2016/6/2'


def parse_time(data):
	DD = datetime.strptime(data, "%Y-%m-%d")
	day = DD.day
	month = DD.month

	return day, month


def SH(data):
	if data == "0":
		return "0"
	else:
		return "1"


dtr = pandas.read_csv("../input/train.csv")
dte = pandas.read_csv("../input/test.csv")

# Open
dte['Open'].fillna(1, inplace=True)

# Dates
dtr['Day'], dtr['Month'] = zip(*dtr['Date'].apply(parse_time))
dte['Day'], dte['Month'] = zip(*dte['Date'].apply(parse_time))

# StateHoliday
dtr['SH'] = dtr['StateHoliday'].apply(SH)
dte['SH'] = dte['StateHoliday'].apply(SH)

dtr.drop(['Date', 'Customers', 'StateHoliday'], inplace=True, axis=1)
dte.drop(['Date', 'StateHoliday'], inplace=True, axis=1)

dtr.to_csv("../input/train2.csv", index=False)
dte.to_csv("../input/test2.csv", index=False)
