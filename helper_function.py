"""
function for final capstone project.
"""
# <h1 style="color:brown;">  Introduction to neural network</h1>

# import data anlysis packages:
import numpy as np
from math import sqrt

# import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error


 
    
def batch_generator(batch_size, sequence_length, num_x_signals=8, num_y_signals=1,
                    x_scaled_train=None, y_scaled_train=None, num_train=None):
    """
    Generator function for creating random batches of training-data.
    Arguments:
        batch_size: batch size wanted to create
        sequence_length: length of sequence we want to shift our sequential data
        num_x_signals: number of imput signals
        and training feature and target sets
    Returns:
        generate batched input 
    """
    Boolean

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

def calculate_errors(train=True, model=None, x_scaled_train=None, x_scaled_test=None,
                    y_train=None, y_test=None, y_scaler=None, target_names=None):
    """
    Calculate traning errors.
    Arguments:
        train: boolean whether the input is training set and test set
        model: training model
        and training feature and target sets
    Returns:
        calculated model errors
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
    
    for signal in range(len(target_names)):
        signal_pred = y_pred_rescaled[:, signal]
        signal_true = y_true[:, signal]
        
        print('    Mean Absolute Error:', mean_absolute_error(signal_true, signal_pred))
        print('    Mean Absolute Percent Error:', mean_absolute_error(signal_true, signal_pred)/np.mean(y_true))
        print('    Root Mean Squared Error:',np.sqrt(mean_squared_error(signal_true, signal_pred)))
        
        
