Crime
================
预测犯罪类别<br>
提交每个类别概率<br>
Xgboost成绩最好，2.35461<br>
logisticR 2.67774<br>
randomForest 2.51186<br>

xgboost
----------------
##### 变量： 
> Category用preprocessing.labelEncoder()转成int型有序变量<br>
> 时间抽出年月日时分,分钟根据时间对称性abs(m-30)<br>
地址判断是否包含'/'<br>
XY进行标准化<br>
区域和DOW变为哑变量 <br>

##### xgboost参数: 
>"booster": "gbtree", "objective": "multi:softprob", "num_class": 39, "max_delta_step": 1, "max_depth": 6<br>

randomForest
--------------------
##### 变量：
> StreetNo第一个字符串是否为数字，Address第二个字符串是否为数字<br>
> 时间抽出小时hour，判断作案是否为晚上dark<br>
> 'Dates', 'DayOfWeek', 'PdDistrict', 'Address'先进行转换，转为有序int型LabelEncoder()<br>
> 'Dates', 'hour', 'dark', 'DayOfWeek', 'PdDistrict', 'StreetNo', 'Address', 'X', 'Y'进行标准化StandardScaler<br>

##### randomForest参数：
> RandomForestClassifier(max_depth=6, n_estimators=50)<br>

logistic Regression
------------------------
##### 变量：
> 区域和DOW哑变量变换<br>
Dates抽出年月日时<br>
XY进行标准变换MinMaxScaler()<br>

##### LR参数：
> LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', random_state=222, C=0.01)<br>

