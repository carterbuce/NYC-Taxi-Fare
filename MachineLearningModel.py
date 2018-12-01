# Python 3
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential, optimizers, regularizers, backend
from sklearn import preprocessing
from tensorflow.python.keras.layers import BatchNormalization, Dense
from Util import save_submission, add_travel_vector_features

# Model parameters
BATCH_SIZE = 250 # how large of batches to run through, larger leads to generalization, smaller leads to overfitting
EPOCHS = 15  # how many times to run all the data through the model
LEARNING_RATE = 0.001

train_df = pd.read_csv('input/train_clean.csv')
test_df = pd.read_csv('input/test.csv')

train_df = train_df.iloc[:1_000_000] #take 500k rows randomly for faster testing TODO remove

# add more features
add_travel_vector_features(test_df)  # note: travel vector features are already in train_df

features = ['abs_diff_distance']

dropped_columns = ['pickup_longitude', 'pickup_latitude',
                   'dropoff_longitude', 'dropoff_latitude',
                   'passenger_count', 'pickup_datetime']
# train_clean = train_df.drop(dropped_columns, axis=1)
# test_clean = test_df.drop(dropped_columns + ['key'], axis=1)

train_features = train_df[features]
test_features = test_df[features]
train_labels = train_df['fare_amount'].values


# Scale data so each feature is between 0 and 1
scaler = preprocessing.MinMaxScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

# create layers of our neural network
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=train_scaled.shape[1], activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))

adam = optimizers.adam(lr=LEARNING_RATE)
model.compile(loss='mse', optimizer=adam, metrics=['mae'])  # mae == rmse

history = model.fit(x=train_scaled, y=train_labels,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=2, validation_split=0.10,
                    shuffle=True)

prediction = model.predict(test_scaled, batch_size=64, verbose=1)  # batch size here doesn't matter

# save our predictions to submission.csv
save_submission(test_df, prediction, 'key', 'fare_amount', 'submission.csv')
