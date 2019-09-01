import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
from statsmodels.tsa.stattools import adfuller

# Input Data (Handle dates)
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m') #Pandas default datetime is YYY-MM-DD-HH:MM:SS
df = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month', date_parser=dateparse) #Convert dataframe into series
ts = df['#Passengers'] #Series now

def test_stationarity(ts):
    print('Dickey-Fuller Results: ')
    # Rolling Stats
    rolmean = ts.rolling(12).mean()
    rolstd = ts.rolling(12).std()

    # Plot rolling statistics
    orig = plt.plot(ts, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label= 'Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Statistics')
    plt.show(block=False)

    # Dikey-Fuller Test:
    dftest = adfuller(ts, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value ({})'.format(key)] = value
    print(dfoutput)


# Make stationary series
# Eliminate Trend (with Smoothing)
ts_log = np.log(ts)
#moving_avg = ts_log.rolling(window=12).mean()
#ts_log_moving_avg_diff.dropna(inplace=True)
expwighted_avg = ts_log.ewm(halflife=12).mean()
ts_log_exp_avg_diff = ts_log - expwighted_avg

# Eliminate Trend and Seasonality (with first order Differencing)
ts_log_diff = ts_log - ts_log.shift()


# Eliminate Trend and Seasonality (with Decomposing)
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_log)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
'''
plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
'''
ts_log_decompose = residual
#ts_log_decompose.dropna(inplace=True)

# Forecasting
'''
# ACF and PACF plots:
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(ts_log_diff, nlags=20)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#Shows p=2 (Intercept of PACF with upper confidence interval), q=2 (Intercept of ACF with upper confidence interval)
'''

# Make ARIMA Model
#AR
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(ts_log, order=(2, 1, 0))
results_AR = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: {:.4f}'.format(sum((results_AR.fittedvalues-ts_log_diff)**2)))

#MA
model = ARIMA(ts_log, order=(0, 1, 2))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log_diff)**2))

#Combined
model = ARIMA(ts_log, order=(2, 1, 2))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
#Significantly lower RSS

# Rescale for predictions
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True) #Note first element missing
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum() #Find cumulative sum
predictions_ARIMA_log = pd.Series(ts_log.ix[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0) #Untransform
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(ts)
plt.plot(predictions_ARIMA)
plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
plt.show()


