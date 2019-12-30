"""
Module provides functions for manipulation and testing of data for the time series analysis
"""
# import data analysis packages
import pandas as pd
import matplotlib.pyplot as plt

# import statsmodels
from statsmodels.tsa.stattools import adfuller

# import hlper functions
import visualization as vis
import helper_functions as f


def test_stationarity(timeseries, window=30):
    """Plot stationary and return Dickey-Fullertest."""
    # determing rolling statistics
    rolmean = timeseries.rolling(window=window, center=False).mean()
    rolstd = timeseries.rolling(window=window, center=False).std()

    # plot rolling statistics
    fig = plt.figure(figsize=(14, 5))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic',
                                'p-value',
                                '#Lags Used',
                                'Number of Observations Used'])
    # print output
    print(dfoutput)
    print('Critical Values:')
    for key, value in dftest[4].items():
        print('\t%s: %0.3f' % (key, value))


def stationarity_autocorrelation_test_original(data, col='chilledwater'):
    """Test stationarity and autocorrelation of time series."""
    print('The purpose of this test is to determine the stationarity and '
          'autocorrelation\n of the original time series.')
    test_stationarity(data, window=7,
                      col='chilledwater')
    vis.acf_pacf(data)


def stationarity_autocorrelation_test_first_diff(data):
    """Test stationarity and autocorrelation of first difference."""
    print('The purpose of this test is to determine the stationarity and \
    autocorrelation of the first difference.')
    first_diff = f.order_difference(data, col='chilledwater')
    test_stationarity(first_diff, window=7, col='chilledwater')
    vis.acf_pacf(first_diff)


def augmented_dickey_fuller_statistics(time_series):
    """Run the augmented Dickey-Fuller test on a time series to determine.
    if it's stationary.
    """
    """
    Arguments:
        time_series: series. Time series that we want to test
    Outputs:
        Test statistics for the Augmented Dickey Fuller test in
        the console
    """

    result = adfuller(time_series.values)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

# def stationarity_autocorrelation_test_second_diff(data):
#     """Test stationarity and autocorrelation of second difference."""
#     print('The purpose of this test is to determine the stationarity and '
#           'autocorrelation\n of the second difference.')
#     first_diff = f.order_difference(data)
#     second_diff = f.order_difference(first_diff)
#     test_stationarity(second_diff['count'], window=12,
#                       title="Second Order Difference")
#     vis1.acf_pacf(second_diff)
