# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

dates = ['2018_08_11',
		 '2018_08_12',
		 '2018_08_13',
		 '2018_08_14',
	]

def replace(file):
	''' Replaces the '[' and ']' with ',' in the txt raw files '''
	f = open(file,'r')
	filedata = f.read()
	f.close()
	newdata = filedata.replace('[', ',')
	newdata = newdata.replace(']', ',')
	f = open(file,'w')
	f.write(newdata)
	f.close()


def Process(dates=dates, city='Cirebon'):
	''' Takes as inputs:
				- the name of a city
				- the set of dates you want to process
		Save a csv file in the local folder that has as index all the links ids,
		as columns all the timestamps and is filled with corresponding speed values.
	'''

	for date in dates:
		Ends = []
		print(date)
		speed_file = '/mnt/divya_storage/Cirebon/CurrentSpeed/'+date+'_CurrentSpeed_Cirebon.txt'
		ID_file = '/mnt/divya_storage/Cirebon/PlaceID/'+date+'_PlaceID_Cirebon.txt'
		# You might have to change this, I worked in a folder Multicity and then two subfolders Speed and ID that contained the files
		replace(speed_file)
		replace(ID_file)
		SpeedData = pd.read_csv(
			speed_file,
			sep=',',
			header=None,
			index_col=False,
			skipinitialspace=True,
			names=np.arange(200000)
			)
		print('------- Speed Data read')

		IdData = pd.read_csv(
			ID_file,
			sep=',',
			header=None,
			index_col=False,
			skipinitialspace=True,
			names=np.arange(200000)
			)
		print('------- ID Data read')

		# You might have to change the names: if your network has more than 200,000 links, it will fails, you just have to have a number larger than the number of links
		# You can also work with try/except to optimise this (if there are far less than 200,000 links, reduce it also to save time)
		SpeedData[0] = SpeedData[0].apply(lambda x: x[:16])
		SpeedData.dropna(axis=1, how='all', inplace=True)
		SpeedData.set_index(keys=0, inplace=True)
		print(len(SpeedData.columns))	# This is the number of links in the network
		IdData[0] = IdData[0].apply(lambda x: x[:16])

		IdData.dropna(axis=1, how='all', inplace=True)
		IdData.set_index(keys=0, inplace=True)
		if len(IdData.index)!=len(SpeedData.index):
			print ('index lengths not matching: '+str(len(IdData.index))+' and '+str(len(SpeedData.index)))	# This means that timestamps are not the same in ID and speed files
		IdData = IdData.transpose()
		Agg = []
		for index, row in SpeedData.iterrows():      # Iterrate over speed data file timestamps, look for the same timestamp in the ID file and store the mapping of both in Agg
			if index in IdData.columns:
				Idrow = IdData[index]
			else:
				Idrow = prev_Idrow
			Idrow = Idrow.rename('ID')
			tp = pd.concat([row, Idrow], axis=1)
			tp.dropna(axis=0, how='all', inplace=True)
			tp.set_index(keys='ID', inplace=True)
			Agg.append(tp)
			prev_Idrow = Idrow

		End = pd.concat(Agg, axis=1)	# Concatenate all the timestamp in one big tab for a date
		Ends.append(End)
		print (Ends)
		to_save = pd.concat(Ends, axis=1)	# Concatenate all the dates in one bigger tab
		to_save.to_csv('/mnt/divya_storage/Cirebon/Processed/'+date+city+'.csv')	  # Save that tab

Process(dates=dates, city='Cirebon')

a=pd.read_excel('/Users/viswak/Downloads/final 1 321.xls')
names = ['Number of lanes','Area','Population','Population density','Speed limit','FreeFlow Speed','average speed (entire day)','average speed generally during the accident time','Drop in Speed (as a percentage to  free flow)','Drop in Speed (as a percentage to  average speed)','Recovery','rate of change of drop']
data1=a[names]
data=a

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data= pd.DataFrame(scaler.fit_transform(data1), columns=data1.columns)
# Import function to create training and test set splits
from sklearn.cross_validation import train_test_split
# Import function to automatically create polynomial features!
from sklearn.preprocessing import PolynomialFeatures
# Import Linear Regression and a regularized regression function
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
# Finally, import function to make a machine learning pipeline
from sklearn.pipeline import make_pipeline
# Alpha (regularization strength) of LASSO regression
lasso_eps = 0.00001
lasso_nalpha=20
lasso_iter=10000
# Min and max degree of polynomials features to consider
degree_min = 2
degree_max = 3
# Test/train split
feature_cols=['average speed generally during the accident time']
X=data[feature_cols]
y=data['Drop in Speed (as a percentage to  average speed)']
X_train,X_test,y_train,y_test=train_test_split(X,y)
# Make a pipeline model with polynomial transformation and LASSO regression with cross-validation, run it for increasing degree of polynomial (complexity of the model)
print len(X_train)
print len(y_train)

for degree in range(degree_min,degree_max+1):
    model = make_pipeline(PolynomialFeatures(degree, interaction_only=False), LassoCV(eps=lasso_eps,n_alphas=lasso_nalpha,max_iter=lasso_iter,
normalize=True,cv=5))
    model.fit(X_train,y_train)
    test_pred = np.array(model.predict(X_test))
    RMSE=np.sqrt(np.sum(np.square(test_pred-y_test)))
    test_score = model.score(X_test,y_test)
#data=data1.loc[data1['Suburb']== 'Mascot']
print test_score
'''
correlations = data.corr()
# plot correlation matrix
print (correlations)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,7,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
'''
'''
X=a[['Speed limit','average speed (entire day)','average speed generally during the accident time','Drop in Speed (as a percentage to  average speed)','Drop in Speed (as a percentage to  free flow)','Drop in Speed (as a percentage to  average speed)']]
y=a['FreeFlow Speed']
# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
print model.summary()
'''
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data= pd.DataFrame(scaler.fit_transform(data1), columns=data1.columns)
print (data)
po=data.std(axis=0)
po1=data.mean(axis=0)
print po
print po1
feature_cols=['average speed (entire day)','FreeFlow Speed']
X=data[feature_cols]
y=data['average speed generally during the accident time']
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,y_train)
print(linreg.intercept_)
print (linreg.coef_)
print zip(feature_cols,linreg.coef_)
y_pred=linreg.predict(X_test)
print('Score: ', linreg.score(X_test, y_test))
from sklearn import metrics
print(metrics.mean_absolute_error(y_test,y_pred))
print(metrics.mean_squared_error(y_test,y_pred))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
