# Python 3
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to


# train_df = pd.read_csv('input/train.csv', nrows=10_000_000)
train_df = pd.read_csv('input/train.csv', nrows=5_000_000, usecols=['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], dtype={'fare_amount': float, 'pickup_longitude': float, 'pickup_latitude': float, 'dropoff_longitude': float, 'dropoff_latitude': float})

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


# remove data with NaN values
print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))

# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train_df)

# exclude data with large differences in latitude/longitude
plot = train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
# plot.figure.show()  # show the plot


print('Old size: %d' % len(train_df))
train_df = train_df[(train_df.abs_diff_longitude < 5.0) & (train_df.abs_diff_latitude < 5.0)]
print('New size: %d' % len(train_df))

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


