"""This module will help us plot a visualizations."""

# import matplotlib
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import statsmodels.api as sm
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


def median_meter_reading(df):
    meter_mapping = {0: 'Electricity', 1: 'Chilled water', 2: 'Steam', 3: 'Hot water'}
    df['meter_type'] = df['meter'].map(meter_mapping)
    
    df.groupby(['timestamp', 'meter_type'])['meter_reading'].median().reset_index().set_index('timestamp') \
    .groupby('meter_type')['meter_reading'] \
    .plot(figsize=(16, 10), title='Meter Reading by Meter Type [kWh]')
    plt.legend()
    plt.ylabel('kWh', fontsize=16)
    plt.savefig('image/meter_reading.png')
    plt.show()


def plot_comparison(start_idx,length=100, train=True, model=None):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """
    
    if train:
        # Use training-data.
        x = x_scaled_train
        y_true = y_train
    else:
        # Use test-data.
        x = x_scaled_test
        y_true = y_test
    
    # End-index for the sequences.
    end_idx = start_idx + length
    
    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]
    
    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    
    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred_rescaled[:, signal]
        
        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15,5))
        
        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        
        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        
        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
#         plt.savefig('image/')
        plt.show()

# def decompose_time_series(series):
#     """
#     Decompose a time series and plot it in the console
#     Arguments:
#         series: series. Time series that we want to decompose
#     Outputs:
#         Decomposition plot in the console
#     """
#     result = seasonal_decompose(series, model='additive')  # multiplicative'
#     result.plot()
#     pyplot.show()


def rolling_statistics(timeseries, window=7, ylabel='Chilled Water'):
    """Plot Rolling mean."""
    rolmean = timeseries.rolling(window=window, center=False).mean()
    rolstd = timeseries.rolling(window=window, center=False).std()

    # plot rolling statistics
    fig = plt.figure(figsize=(12, 6))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.xlabel('Year', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

def seasonal_decompose_plot(data_logscaled):
    """Plot seasonal decomposition."""
    decomposition = seasonal_decompose(data_logscaled)  # , freq=12)

    # Gather the trend, seasonality and noise of decomposed object
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot gathered statistics
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(data_logscaled, label='Original', color="blue")
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend', color="blue")
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality', color="blue")
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals', color="blue")
    plt.legend(loc='best')
    plt.tight_layout()

    # check stationaryity of residuals using stationarity_check
    # ts_log_decompose = residual
    # ts_log_decompose.dropna(inplace=True)

    # Check stationarity
    # stationarity_check(ts_log_decompose)
    return residual


def test_stationarity(data, window=12, col='electricity'):
    """
    Creates stationarity plot, returns results of Dickey-Fuller test
    """
    rolling_statistics(data, window=12, col='electricity')
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(data[col], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4],
                         index=['Test Statistic',
                                'p-value',
                                '#Lags Used',
                                'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    
    print(dfoutput)
    plt.savefig('image/stationarity_plot_chw.png')
    
    
def acf_pacf(data, col='electricity'):
    """
    Plots autocorrelation function (ACF) and partial autocorrelation function
    (PACF) outputs
    """
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data[col].iloc[1:], lags=13, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data[col].iloc[1:], lags=13, ax=ax2) 

    
def stationarity_autocorrelation_test_original(data, col='electricity', title="Original Time series"):
    """
    Tests stationarity and autocorrelation of time series
    """
    print('The purpose of this test is to determine the stationarity and '
          'autocorrelation\n of the original time series.')
    test_stationarity(data, window=12)
    acf_pacf(data)

    
def tsplot(y, title, lags=None, figsize=(12,8)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax =   plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax =  plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    
    y.plot(ax = ts_ax)
    ts_ax.set_title(title, fontsize=16, fontweight='bold')
    y.plot(ax=hist_ax, kind='hist', bins=25)
    hist_ax.set_title('Histogram')
    sm.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax)
    sm.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax)
#     [ax.set_xlim[0] for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    plt.savefig('image/ts_plot_chw.png')

    return ts_ax, acf_ax, pacf_ax
    
# """ 
# Pyplot."""
# # Make subplot figure
# fig = make_subplots(specs=[[{"secondary_y": True}]])
# # Add traces; weekday and weekend aggregate rides
# fig.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['ride_count'], name="Actual Number of Rides Each Hour",
#                          line_color='red'))
# fig.add_trace(go.Scatter(x=y_test_df.index, y=y_test_df['circular_rf'], name="Predicted Rides Each Hour",
#                          line_color='black'), secondary_y=False)
# fig.update_layout(title_text='Predicted and Actual Rides Each Hour')
# # Set x-axis title
# fig.update_xaxes(title_text="Hour of Day")
# # Set y-axes titles
# fig.update_yaxes(title_text="<b>Numer of Rides</b> initiated", secondary_y=False)
# fig.update_yaxes(title_text="<b>Rides</b>", secondary_y=False)
# # Include x-axis slider
# fig.update_layout(xaxis_rangeslider_visible=True)
# fig.show()


def plot_train_vs_val_loss(history=None, a=None):
    """
    plot model accuracy, comparing training and testing accuracies.
    Arguments:
        history: model history 
    Outputs:
        MSE loss plot 
    """
    fig = go.Figure()
#     fig.add_trace(go.Scatter(y=history.history['loss'], name="Training Loss"))
    fig.add_trace(go.Scatter(y=history.history['val_loss'], name="validation Loss"))
    fig.update_layout(
                      title= {'text': "Training vs validation loss ({})".format(a),
                              'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},
                      xaxis_title="Epochs",
                      yaxis_title="Loss",
                      font=dict(family="Courier New, monospace", size=16, color='black'),
                     )
#     width=1000, height=500
#     fig.write_image('image/train2.png')
#     fig['layout']['yaxis1'].update(range=[0, 0.05], autorange=False)
    fig.show();

