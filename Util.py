import numpy as np
import pandas as pd

# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    df['abs_diff_distance'] = df.abs_diff_latitude + df.abs_diff_longitude # manhattan distance
    # df['abs_diff_distance'] = np.sqrt(np.power(df.abs_diff_longitude, 2) + np.power(df.abs_diff_latitude, 2))


def save_submission(raw_test, prediction, id_column, prediction_column, file_name):
    df = pd.DataFrame(prediction, columns=[prediction_column])
    df[id_column] = raw_test[id_column]
    df[[id_column, prediction_column]].to_csv((file_name), index=False)
    print('submission.csv saved')


