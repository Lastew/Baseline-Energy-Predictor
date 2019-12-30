"""This module will help us plot a visualizations."""

# import data anlysis packages:
import numpy as np

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


# gcloud compute ssh jupyter@stikini-1-nogpu --zone us-central1-b --project final-project-2019-12-05 -- -L 8080:localhost:8080

def plot_comparison(start_idx, length=100, train=True, model=None,
                    x_train_scaled=None, x_test_scaled=None,
                    y_train=None, y_test=None,
                    y_scaler=None, target_names=None, warmup_steps=None):
    """
    Plot the predicted and true output-signals.
    Arguments
        start_idx: Start-index for the time-series.
        length: Sequence-length to process and plot.
        train: Boolean whether to use training- or test-set.
        model: The model we want to plot
    Returns
        comparison plot
    """

    if train:
        # Use training-data.
        x = x_train_scaled
        y_true = y_train
    else:
        # Use test-data.
        x = x_test_scaled
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
        plt.figure(figsize=(15, 5))

        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')

        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()
