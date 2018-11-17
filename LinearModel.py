# Python 3
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to


# train_df = pd.read_csv('input/train.csv', nrows=10_000_000)
train_df = pd.read_csv('input/train.csv', usecols=['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count'],
                       dtype={'fare_amount': float, 'pickup_longitude': float, 'pickup_latitude': float, 'dropoff_longitude': float, 'dropoff_latitude': float, 'passenger_count': int})

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


# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    df['abs_diff_distance'] = np.sqrt(np.power(df.abs_diff_longitude, 2) + np.power(df.abs_diff_latitude, 2))

add_travel_vector_features(train_df)


# remove data with NaN values
print('Original size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size after removing NaN location values: %d' % len(train_df))

# remove data with negative and large fares
train_df = train_df[(train_df.fare_amount > 0) & (train_df.fare_amount < 500)]
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


# train_df = train_df[(train_df.abs_diff_longitude < 3.0) & (train_df.abs_diff_latitude < 3.0)]
# train_df = train_df[(train_df.abs_diff_distance < 3.0)]
# print('New size after removing abs diff distance >3.0 degrees: %d' % len(train_df))


# plot = train_df.plot.scatter('abs_diff_distance', 'fare_amount')
# plot.figure.show()
#
# plot = train_df.plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
# plot.figure.show()
#
# plot = train_df.plot.scatter('dropoff_longitude', 'dropoff_latitude')
# plot.figure.show()
#
# plot = train_df.plot.scatter('pickup_longitude', 'pickup_latitude')
# plot.figure.show()


# Construct and return an Nx3 input matrix for our linear model
# using the travel vector, plus a 1.0 for a constant bias term.
def get_input_matrix(df):
    return np.column_stack((df.abs_diff_longitude, df.abs_diff_latitude, np.ones(len(df))))

train_X = get_input_matrix(train_df)
train_y = np.array(train_df['fare_amount'])

print(train_X.shape)
print(train_y.shape)

# The lstsq function returns several things, and we only care about the actual weight vector w.
(w, _, _, _) = np.linalg.lstsq(train_X, train_y, rcond = None)
print(w)

test_df = pd.read_csv('input/test.csv')
# test_df.dtypes:
# key                   object
# pickup_datetime       object
# pickup_longitude     float64
# pickup_latitude      float64
# dropoff_longitude    float64
# dropoff_latitude     float64
# passenger_count        int64
# dtype: object

# Reuse the above helper functions to add our features and generate the input matrix.
add_travel_vector_features(test_df)
test_X = get_input_matrix(test_df)
# Predict fare_amount on the test set using our model (w) trained on the training set.
test_y_predictions = np.matmul(test_X, w).round(decimals = 2)

# Write the predictions to a CSV file which we can submit to the competition.
submission = pd.DataFrame(
    {'key': test_df.key, 'fare_amount': test_y_predictions},
    columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)


