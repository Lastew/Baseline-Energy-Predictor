"""Boolian masking."""
import numpy as np
import math

# no_rain = spark_df.filter(spark_df['rain'] == 0.0)

# print("Training Dataset Count: " + str(trainingData.count()))
# print("Test Dataset Count: " + str(testData.count()))

# ad900d


def fill_nan(df, cols=['floor_count', 'year_built']):
    """Fill the nan column values with '-999'."""
    for col in cols:
        df[col] = df[col].fillna(-999).astype(np.int16)

    return df


def split_timestamp(df):
    """Split the timestamp to day, day of the week and hour."""
    df['month_datetime'] = df['timestamp'].dt.month.astype('uint8')
    df['weekofyear_datetime'] = df['timestamp'].dt.weekofyear.astype('uint8')
    df['dayofyear_datetime'] = df['timestamp'].dt.dayofyear.astype('uint8')

    df['hour_datetime'] = df['timestamp'].dt.hour.astype('uint8')
    df['day_week'] = df['timestamp'].dt.dayofweek.astype('uint8')
    df['day_month_datetime'] = df['timestamp'].dt.day.astype('uint8')
    df['week_month_datetime'] = df['timestamp'].dt.day/7
    df['week_month_datetime'] = df['week_month_datetime'].apply(
        lambda x: math.ceil(x)).astype(np.int8)

    return df


# def sisay(df, cols=['year_built', 'square_feet']):
#     """Convert the built year and square feet columns."""
#     if df.columns == 'year_built'
#     for col in cols:
#         if col = 'year_built':
#             df[col] = df[col]-1900
#         elif col = 'square_feet':
#             df[col] = np.log(df[col])
#     return df
