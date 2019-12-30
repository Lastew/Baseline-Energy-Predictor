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
    """
    Reduces the memory usage of the original dataset.
    Arguments:
        df: original sized dataframe
    Outputs:
        Reduced memory usage dataset
    """
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
    """
    Split timestamp to month, day_of_month, day_of_week _series.
    Arguments:
        df: original sized dataframe
    Outputs:
        Splited timestamp dataset
    """
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
    Arguments
        df: data frame.
        colname: string. Name of to be valued column
    Returns
        df_count: data frame.
    """
    df_count = df[colname].value_counts().to_frame().reset_index()
    df_count = df_count.rename(columns={'index': colname+'_values',
                                        colname: 'counts'})
    return df_count


def check_missing(df, cols=None, axis=0):
    """Check for missing values
    Arguments
        df: dataframe
        cols: list. List of column names
        axis: int. 0 means column and 1 means row
    Returns
        missing_info: data frame
    """
    if cols != None:
        df = df[cols]
    missing_num = df.isnull().sum(axis).to_frame().rename(columns={0: 'missing_num'})
    missing_num['missing_percent'] = df.isnull().mean(axis)*100
    return missing_num.sort_values(by='missing_percent', ascending=False)


def extract_data(building_id, train_data=None):
    """Extract data that is going to be used by ML model.
    Arguments
        building_id: duilding id number
        train_data: dataset
    Returns
        ML model ready dataset
        """
    building_data = train_data[train_data['building_id'] == building_id].drop(['x', 'y'])

    building_data.ffill()
    building_data.bfill()
