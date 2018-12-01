# Python 3
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to
from Util import *


# train_df = pd.read_csv('input/train.csv', nrows=10_000_000)

columns = ['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
column_types = {'fare_amount': float,
                'pickup_datetime': str,
                'pickup_longitude': float,
                'pickup_latitude': float,
                'dropoff_longitude': float,
                'dropoff_latitude': float,
                'passenger_count': int
                }

train_df = pd.read_csv('input/train.csv', usecols=columns, dtype=column_types)
# train_df = pd.read_csv('input/train.csv')

# train_df.dtypes:
# key                   object
# fare_amount          float64
# pickup_datetime       object
# pickup_longitude     float64
# pickup_latitude      float64
# dropoff_longitude    float64
# dropoff_latitude     float64
# passenger_count        int64
# dtype: object



add_travel_vector_features(train_df)

# remove data with NaN values
print('Original size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size after removing NaN location values: %d' % len(train_df))

# remove data with negative and large fares
train_df = train_df[(train_df.fare_amount > 0) & (train_df.fare_amount < 250)]
print('New size after removing negative and outlier fares: %d' % len(train_df))

# remove data with locations outside of NYC
train_df = train_df[(train_df.dropoff_longitude < -72.0) & (train_df.dropoff_longitude > -75.0)]
train_df = train_df[(train_df.dropoff_latitude < 42.0) & (train_df.dropoff_latitude > 40.0)]
train_df = train_df[(train_df.pickup_longitude < -72.0) & (train_df.pickup_longitude > -75.0)]
train_df = train_df[(train_df.pickup_latitude < 42.0) & (train_df.pickup_latitude > 40.0)]
train_df = train_df[(train_df.abs_diff_distance > 0)]
print('New size after restricting pickup/dropoff lat/long: %d' % len(train_df))

# remove data with more than 4 passengers
train_df = train_df[(train_df.passenger_count < 5)]
print('New size after removing passenger counts > 4: %d' % len(train_df))

train_df.to_csv('input/train_clean.csv', index=False)