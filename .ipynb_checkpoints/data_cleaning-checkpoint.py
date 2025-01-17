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
