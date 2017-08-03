import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader as web
import bs4 as bs
import pickle
import requests
import time
style.use('ggplot')

def fetchData(tickers):
	print("Fetching data from Yahoo Finances.. (2 min aprox)")
	resp = requests.get('https://es-us.finanzas.yahoo.com/quote/%5EMERV/components?p=%5EMERV')
	soup = bs.BeautifulSoup(resp.text,'lxml')
	table = soup.find(id="Main")
	for row in table.findAll('tr')[1:]: 
		ticker = row.findAll('td')[0].text
		tickers.append(ticker)
		



tickers = []
start = dt.datetime(2017,7,1)
end = dt.datetime(2017,8,2)
fetchData(tickers)
for ticker in tickers:
	time.sleep(3)
	df = web.DataReader(ticker,'yahoo',start,end)
	df.to_csv(ticker+'.csv')
	print(ticker + " saved.")
#df = web.DataReader('EDN.BA','yahoo',start,end)
#df.to_csv('EDN.csv')
#df = pd.read_csv('EDN.csv', parse_dates=True, index_col=0)
#df['10ma'] = df['Adj Close'].rolling(window=10,min_periods=0).mean()
#df['20ma'] = df['Adj Close'].rolling(window=20,min_periods=0).mean()
#ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
#ax1.plot(df.index, df['Adj Close'])
#ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

#ax1.plot(df.index, df['10ma'])
#ax1.plot(df.index, df['20ma'])
#ax2.bar(df.index,df['Volume'])

#plt.show()