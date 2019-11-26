"""
function for final capstone project.
"""
# <h1 style="color:brown;">  Introduction to neural network</h1>

# import data anlysis packages:
import numpy as np
from math import sqrt

# import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error


def reduce_mem_usage(df, verbose=True):
    """Reduces the size of our dataframe."""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem)/start_mem))
    return df


def calculate_model_accuracy_metrics(actual, predicted):
    """
    Output model accuracy metrics, comparing predicted values
    to actual values.
    Arguments:
        actual: list. Time series of actual values.
        predicted: list. Time series of predicted values
    Outputs:
        Forecast bias metrics, mean absolute error, mean squared error,
        and root mean squared error in the console
    """
    # Calculate forecast bias
    forecast_errors = [actual[i]-predicted[i] for i in range(len(actual))]
    bias = sum(forecast_errors) * 1.0/len(actual)
    print('Bias: %f' % bias)
    # Calculate mean absolute error
    mae = mean_absolute_error(actual, predicted)
    print('MAE: %f' % mae)
    # Calculate mean squared error and root mean squared error
    mse = mean_squared_error(actual, predicted)
    print('MSE: %f' % mse)
    rmse = sqrt(mse)
    print('RMSE: %f' % rmse)


def split_timestamp(df):
    """Split time_series."""
    df['month'] = df['timestamp'].dt.month.astype('uint8')
    df['day_of_month'] = df['timestamp'].dt.day.astype('uint8')
    df['day_of_week'] = df['timestamp'].dt.dayofweek.astype('uint8')
    df['hour'] = df['timestamp'].dt.hour.astype('uint8')

    return df


def order_difference(data, col='chilledwater'):
    """Create dataset with order difference."""
    data_diff = data.copy()
    data_diff[col] = data[col].diff()
    data_diff.dropna(inplace=True)
    return data_diff


def feat_value_count(df, colname):
    """Value count of each feature

    Args
    df: data frame.
    colname: string. Name of to be valued column

    Returns
    df_count: data frame.
    """
    df_count = df[colname].value_counts().to_frame().reset_index()
    df_count = df_count.rename(columns={'index': colname+'_values', colname: 'counts'})
    return df_count


def feat_value_count(df, colname):
    """Count the value of each
    Args
    df: data frame.
    colname: string. Name of to be valued column

    Returns
    df_count: data frame.
    """
    df_count = df[colname].value_counts().to_frame().reset_index()
    df_count = df_count.rename(columns={'index': colname+'_values', colname: 'counts'})
    return df_count


def check_missing(df, cols=None, axis=0):
    """Check for missing values
    Args
    df: data frame.
    cols: list. List of column names
    axis: int. 0 means column and 1 means row

    Returns
    missing_info: data frame.
    """
    if cols != None:
        df = df[cols]
    missing_num = df.isnull().sum(axis).to_frame().rename(columns={0: 'missing_num'})
    missing_num['missing_percent'] = df.isnull().mean(axis)*100
    return missing_num.sort_values(by='missing_percent', ascending=False)


def plot_comparison(start_idx, model=model,length=100, train=True, x=x_train_scaled, x=x_test_scaled, y_true=y_train, y_true=y_test):
    """
    Plot the predicted and true output-signals.
    
    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
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
        plt.figure(figsize=(15,5))
        
        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')
        
        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)
        
        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()