# Python 3
import numpy as np  # linear algebra
import pandas as pd  # CSV file I/O (e.g. pd.read_csv)
import os  # reading the input files we have access to
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential, optimizers, regularizers
from sklearn import preprocessing
from tensorflow.python.keras.layers import BatchNormalization, Dense
from Util import output_submission, add_travel_vector_features

# Model parameters
BATCH_SIZE = 256
EPOCHS = 20  #50
LEARNING_RATE = 0.001

train_df = pd.read_csv('input/train_clean.csv')
test_df = pd.read_csv('input/test.csv')


# add more features
add_travel_vector_features(test_df) # travel vector features already in train_clean



dropped_columns = ['pickup_longitude', 'pickup_latitude',
                   'dropoff_longitude', 'dropoff_latitude',
                   'passenger_count', 'pickup_datetime']
train_clean = train_df.drop(dropped_columns, axis=1)
test_clean = test_df.drop(dropped_columns + ['key'], axis=1)


train_df, validation_df = train_test_split(train_clean, test_size=0.10, random_state=1)

# Get labels
train_labels = train_df['fare_amount'].values
validation_labels = validation_df['fare_amount'].values
train_df = train_df.drop(['fare_amount'], axis=1)
validation_df = validation_df.drop(['fare_amount'], axis=1)

# Scale data
# Note: im doing this here with sklearn scaler but, on the Coursera code the scaling is done with Dataflow and Tensorflow
scaler = preprocessing.MinMaxScaler()
train_df_scaled = scaler.fit_transform(train_df)
validation_df_scaled = scaler.transform(validation_df)
test_scaled = scaler.transform(test_clean)

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=train_df_scaled.shape[1], activity_regularizer=regularizers.l1(0.01)))
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
model.compile(loss='mse', optimizer=adam, metrics=['mae'])

history = model.fit(x=train_df_scaled, y=train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    verbose=1, validation_data=(validation_df_scaled, validation_labels),
                    shuffle=True)

prediction = model.predict(test_scaled, batch_size=128, verbose=1)

output_submission(test_df, prediction, 'key', 'fare_amount', 'submission.csv')
