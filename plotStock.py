import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader as web
import bs4 as bs
import pickle
import requests
import time
import os
import numpy as np
from collections import Counter
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

style.use('ggplot')

def fetchData():
	print("Fetching data from Yahoo Finances.. (2 min aprox)")
	tickers = []
	resp = requests.get('https://es-us.finanzas.yahoo.com/quote/%5EMERV/components?p=%5EMERV')
	soup = bs.BeautifulSoup(resp.text,'lxml')
	table = soup.find(id="Main")
	for row in table.findAll('tr')[1:]: 
		ticker = row.findAll('td')[0].text
		tickers.append(ticker)
	with open("stocks.pickle","wb") as f:
			pickle.dump(tickers,f)	
	return tickers

def getSavedData(reload=False):
	if reload:
		tickers = fetchData()
	else:
		with open("stocks.pickle","rb") as f:
			tickers = pickle.load(f)
	start = dt.datetime(2017,7,1)
	end = dt.datetime(2017,8,2)
	if not os.path.exists('stocks'):
		os.makedirs('stocks')
	for ticker in tickers:
		if not os.path.exists('stocks/{}.csv'.format(ticker)):
			time.sleep(0.5)
			df = web.DataReader(ticker,'yahoo',start,end)
			df.to_csv('stocks/{}.csv'.format(ticker))
			print(ticker + " saved.")
		else:
			print("Already have {}".format(ticker))
	print()
	print()

def loadData():
	with open("stocks.pickle","rb") as f:
		tickers = pickle.load(f)
	mainDf = pd.DataFrame()
	for count,ticker in enumerate(tickers):
		df = pd.read_csv('stocks/{}.csv'.format(ticker))
		df.set_index('Date', inplace=True)
		df.rename(columns={'Adj Close':ticker}, inplace=True)
		df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
		if mainDf.empty:
			mainDf = df
		else:
			mainDf = mainDf.join(df, how='outer')
		if count % 10 == 0:
			print(count)
	print(mainDf.head())
	mainDf.to_csv('StocksUnidas.csv')

def readData():
	df = pd.read_csv('StocksUnidas.csv')
	dfCorr = df.corr()
	dfCorr.to_csv('StocksCorr.csv')
	data1 = dfCorr.values
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	heatmap = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
	fig.colorbar(heatmap)
	ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
	ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
	ax1.invert_yaxis()
	ax1.xaxis.tick_top()
	column_labels = dfCorr.columns
	row_labels = dfCorr.index
	ax1.set_xticklabels(column_labels)
	ax1.set_yticklabels(row_labels)
	plt.xticks(rotation=90)
	heatmap.set_clim(-1,1)
	plt.tight_layout()
	plt.show()

def processData(ticker):
	hm_days = 7
	df = pd.read_csv('StocksUnidas.csv', index_col=0)
	tickers = df.columns.values.tolist()
	df.fillna(0, inplace=True)
	for i in range(1,hm_days+1):
		df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
	df.fillna(0, inplace=True)
	return tickers, df
def buy_sell_hold(*args):
	cols = [c for c in args]
	requirement = 0.02
	for col in cols:
		if col > requirement:
			return 1
		if col < -requirement:
			return -1
	return 0

def extractFeatures(ticker):
	tickers, df = processData(ticker)
	df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
												df['{}_1d'.format(ticker)],
												df['{}_2d'.format(ticker)],
												df['{}_3d'.format(ticker)],
												df['{}_4d'.format(ticker)],
												df['{}_5d'.format(ticker)],
												df['{}_6d'.format(ticker)],
												df['{}_7d'.format(ticker)] ))
	vals = df['{}_target'.format(ticker)].values.tolist()
	str_vals = [str(i) for i in vals]
	print('Stock', ticker)
	print('Data spread:',Counter(str_vals))
	df.fillna(0, inplace=True)
	df = df.replace([np.inf, -np.inf], np.nan)
	df.dropna(inplace=True)
	#Transformo los valores en porcentajes
	df_vals = df[[ticker for ticker in tickers]].pct_change()
	df_vals = df_vals.replace([np.inf, -np.inf], 0)
	df_vals.fillna(0, inplace=True)
	X = df_vals.values
	y = df['{}_target'.format(ticker)].values
	return X,y,df

def doML(ticker):
	X, y, df = extractFeatures(ticker)
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
														y,
														test_size=0.25)
	clf = VotingClassifier([('lsvc',svm.LinearSVC()),
							('knn',neighbors.KNeighborsClassifier()),
							('rfor',RandomForestClassifier())])
	clf.fit(X_train, y_train)
	confidence = clf.score(X_test, y_test)
	
	print('accuracy:',confidence)
	predictions = clf.predict(X_test)
	print('predicted class counts:',Counter(predictions))
	print()
	print()
tickers = getSavedData(True)
readData()
doML('APBR.BA')
doML('EDN.BA')
doML('BMA.BA')

