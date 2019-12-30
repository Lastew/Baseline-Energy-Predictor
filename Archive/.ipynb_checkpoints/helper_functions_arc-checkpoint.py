"""
function for final capstone project.
"""
# <h1 style="color:brown;">  Introduction to neural network</h1>

# import data anlysis packages:
import numpy as np

# import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_train_vs_val_accuracy(history=None):
    """
    plot model accuracy, comparing training and testing accuracies.
    Arguments:
        history: model history 
    Outputs:
        accuracy plot 
    """
    plt.figure(figsize=(10,5))
    plt.plot(history.history['acc'], label='Train Accuracy')
    plt.plot(history.history['val_acc'], label='Test Accuracy')
    plt.title('Model train vs validation accuracy', fontsize=20)
    plt.ylabel('accuracy', fontsize=16)
    plt.xlabel('epochs', fontsize=16)
    plt.xticks(fontsize=12), plt.yticks(fontsize=12)
    plt.legend(loc='upper right', fontsize=14)
    sns.set_context("paper", font_scale=1.3)
    sns.set_style('white')
    plt.show();
    
    
def plot_train_vs_val_loss(history=None):
    """
    plot model loss, comparing training and testing losses.
    Arguments:
        history: model history 
    Outputs:
        loss plot 
    """
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model train vs validation loss', fontsize=20)
    plt.ylabel('loss', fontsize=16)
    plt.xlabel('epochs', fontsize=16)
    plt.xticks(fontsize=12), plt.yticks(fontsize=12)
    plt.legend(loc='upper right', fontsize=14)
    sns.set_context("paper", font_scale=1.3)
    sns.set_style('white')
    plt.show();    

    
def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            
            # Copy the sequences of data starting at this index.
            x_batch[i] = x_scaled_train[idx:idx+sequence_length]
            y_batch[i] = y_scaled_train[idx:idx+sequence_length]
        
        yield (x_batch, y_batch)
        

# def calculate_model_accuracy_metrics(actual, predicted):
#     """
#     Output model accuracy metrics, comparing predicted values
#     to actual values.
#     Arguments:
#         actual: list. Time series of actual values.
#         predicted: list. Time series of predicted values
#     Outputs:
#         Forecast bias metrics, mean absolute error, mean squared error,
#         and root mean squared error in the console
#     """
#     # Calculate forecast bias
#     forecast_errors = [actual[i]-predicted[i] for i in range(len(actual))]
#     bias = sum(forecast_errors) * 1.0/len(actual)
#     print('Bias: %f' % bias)
#     # Calculate mean absolute error
#     mae = mean_absolute_error(actual, predicted)
#     print('MAE: %f' % mae)
#     # Calculate mean squared error and root mean squared error
#     mse = mean_squared_error(actual, predicted)
#     print('MSE: %f' % mse)
#     rmse = sqrt(mse)
#     print('RMSE: %f' % rmse)
    

def calculate_model_accuracy_metrics(train=True, model=None, x_scaled_train=None, x_scaled_test=None, y_train=None, y_test=None):
    """
    Output model accuracy metrics, comparing predicted values
    to actual values.
    Arguments:
        boolan :
    Outputs:
        Forecast bias metrics, mean absolute error, mean squared error,
        and root mean squared error in the console
    """
    if train:
        # Use training-data.
        x = x_scaled_train
        y_true = y_train
        print('Traning data:')
    else:
        # Use test-data.
        x = x_scaled_test
        y_true = y_test
        print('Testing data:')
        
    x = np.expand_dims(x, axis=0)
    y_pred = model.predict(x)
    y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])
    
    # Calculate mean absolute error, mean squared error and root mean squared error terms
    for signal in range(len(target_names)):
        
        signal_pred = y_pred_rescaled[:, signal]
        signal_true = y_true[:, signal]
        
        mae = mean_absolute_error(signal_true, signal_pred)
        mse = mean_squared_error(signal_true, signal_true)
        rmse = np.sqrt(mse)
        print('    Mean Absolute Error: %f' % mae)
        print('    Root Mean Squared Error: %f' % rmse)
        print('    --------------------------------')
        



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
        
        
def loss_mse_warmup(y_true, y_pred):
   # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(y_true=y_true_slice, y_pred=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean