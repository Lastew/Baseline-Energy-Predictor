"""This module will help us plot a visualizations."""

# import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# import statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose

# Set specific parameters for the visualizations
large = 22
med = 16
small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style('darkgrid')


def decompose_time_series(series):
    """
    Decompose a time series and plot it in the console
    Arguments:
        series: series. Time series that we want to decompose
    Outputs:
        Decomposition plot in the console
    """
    result = seasonal_decompose(series, model='additive')  # multiplicative'
    result.plot()
    pyplot.show()


def rolling_statistics(data, window=7, col='chilledwater'):
    """Plot Rolling mean."""
    rolmean = data[col].rolling(window=window, center=False).mean()
    rolstd = data[col].rolling(window=window, center=False).std()

    # plot rolling statistics
    fig = plt.figure(figsize=(12, 6))
    orig = plt.plot(data[col], color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.xlabel('Year', fontsize=15)
    plt.ylabel(col)
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

# def seasonal_decompose_plot(data_logscaled):
#     """Plot seasonal decomposition."""
#     decomposition = seasonal_decompose(data_logscaled)  # , freq=12)
#
#     # Gather the trend, seasonality and noise of decomposed object
#     trend = decomposition.trend
#     seasonal = decomposition.seasonal
#     residual = decomposition.resid
#
#     # Plot gathered statistics
#     plt.figure(figsize=(12, 8))
#     plt.subplot(411)
#     plt.plot(data_logscaled, label='Original', color="blue")
#     plt.legend(loc='best')
#     plt.subplot(412)
#     plt.plot(trend, label='Trend', color="blue")
#     plt.legend(loc='best')
#     plt.subplot(413)
#     plt.plot(seasonal, label='Seasonality', color="blue")
#     plt.legend(loc='best')
#     plt.subplot(414)
#     plt.plot(residual, label='Residuals', color="blue")
#     plt.legend(loc='best')
#     plt.tight_layout()
#
#     # check stationaryity of residuals using stationarity_check
#     # ts_log_decompose = residual
#     # ts_log_decompose.dropna(inplace=True)
#
#     # Check stationarity
#     # stationarity_check(ts_log_decompose)
#     return residual


""" 
Pyplot."""
# Make subplot figure
fig = make_subplots(specs=[[{"secondary_y": True}]])
# Add traces; weekday and weekend aggregate rides
fig.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['ride_count'], name="Actual Number of Rides Each Hour",
                         line_color='red'))
fig.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['circular_rf'], name="Predicted Rides Each Hour",
                         line_color='black'), secondary_y=False)
fig.update_layout(title_text='Predicted and Actual Rides Each Hour')
# Set x-axis title
fig.update_xaxes(title_text="Hour of Day")
# Set y-axes titles
fig.update_yaxes(title_text="<b>Numer of Rides</b> initiated", secondary_y=False)
fig.update_yaxes(title_text="<b>Rides</b>", secondary_y=False)
# Include x-axis slider
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
