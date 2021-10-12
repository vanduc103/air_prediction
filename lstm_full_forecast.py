### Stacked Fully Connected LSTM for forecasting

from __future__ import print_function

from read_data import read_data

import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

import random
import time
import os

# random number
seed = 128
rng = np.random.RandomState(seed)

def batch_creator(X, batch_size, dataset_length):
    batch_x = list()
    batch_y = list()

    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length - timesteps - pred_timesteps, batch_size)
    for i in range(len(batch_mask)):
        offset = batch_mask[i]
        
        batch_x.append(X[offset : offset + timesteps])
        batch_y.append(X[offset + timesteps : offset + timesteps + pred_timesteps])
    
    batch_x = np.asarray(batch_x)
    batch_y = np.asarray(batch_y)
    batch_ymap = np.zeros((batch_size, output_size))
    batch_ymap[:, station_map] = 1.0

    batch_x = batch_x.reshape((batch_size, timesteps, input_features))
    batch_y = batch_y.reshape((batch_size, output_size))

    return batch_x, batch_y, batch_ymap

# load pollution data
pollution_file = 'data/pollutionPM25.h5'
if os.path.isfile(pollution_file):
    with h5py.File(pollution_file, 'r') as hf:
        X = hf['pollution'][:]
        station_map = hf['station_map'][:]

print(X.shape)

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X.reshape(X.shape[0]*X.shape[1],1)).reshape(X.shape[0], X.shape[1])

# split to train, validate, test set
train_size = (365+366)*24
X_train, X_test = X[:train_size], X[train_size:]
split_size = train_size - (92)*24
X_train, X_val = X_train[:split_size], X_train[split_size:]
print('Training set shape: {}'.format(X_train.shape))
print('Validate set shape: {}'.format(X_val.shape))
print('Test set shape: {}'.format(X_test.shape))

# Training Parameters
timesteps = 12 # timesteps
pred_timesteps = 12 # predict timesteps

learning_rate = 0.001
training_steps = 200
batch_size = 128
display_step = 20
is_training = True

# Network Parameters
grid_size = 1024
input_features = grid_size
n_hidden = 2000 # hidden layer
fc_size = 1000
dr_rate = 0.5
loss_ratio = grid_size/len(station_map)
output_size = grid_size * pred_timesteps

# tf Graph input
x = tf.placeholder("float", [None, timesteps, input_features])

def prediction_model(x):
    # define the model
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_features)
    input_x = tf.unstack(x, timesteps, axis=1)

    # LSTM layers with n_hidden units.
    num_units = [n_hidden, n_hidden, n_hidden]
    cells = [rnn.LSTMCell(num_units=n) for n in num_units]
    rnn_cell = rnn.MultiRNNCell(cells)

    # dropout
    #rnn_cell = rnn.DropoutWrapper(rnn_cell, output_keep_prob=dropout)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, input_x, dtype=tf.float32)

    # Batch Norm
    states_norm = tf.layers.batch_normalization(states[-1][-1], training=is_training)

    # fully connected
    W_fc = tf.get_variable(name='W_fc', shape=[n_hidden, fc_size], 
	        initializer=tf.contrib.layers.xavier_initializer())
    b_fc = tf.Variable(tf.zeros(fc_size))
    fc = tf.nn.relu(tf.add(tf.matmul(states_norm, W_fc), b_fc))

    # Batch Norm
    fc_norm = tf.layers.batch_normalization(fc, training=is_training)

    # l2 regularization
    l2 = tf.nn.l2_loss(W_fc)

    # drop out layer
    dropout = tf.layers.dropout(
        inputs=fc_norm, rate=dr_rate, training=True)

    return dropout, l2

with tf.variable_scope('prediction_model'):
    fc_out, fc_l2 = prediction_model(x)

# output layer
W_output = tf.get_variable(name='W_output', shape=[fc_size, output_size],   
                initializer=tf.contrib.layers.xavier_initializer())
b_output = tf.Variable(tf.zeros(output_size))
# use sigmoid function to make output in (0,1) but not 0
output = tf.nn.sigmoid(tf.add(tf.matmul(fc_out, W_output), b_output))
output_l2 = tf.nn.l2_loss(W_output)

# tf Graph output
y = tf.placeholder("float", [None, output_size])
y_map = tf.placeholder("float", [None, output_size])

if is_training:
    # Loss and optimizer
    yhat = tf.multiply(output, y_map)
    loss = tf.sqrt(tf.losses.mean_squared_error(labels=y, predictions=yhat) * loss_ratio)

    beta = 0.01
    regularizer = output_l2 + fc_l2
    loss = tf.reduce_mean(loss + beta * regularizer)
    tf.summary.scalar('lstm_forecast_loss', loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    # Batch Norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()
model_path = "model/lstm_forecast.ckpt"

# Merge all the summaries and write them out
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/lstm_forecast_train', flush_secs=10)
val_writer = tf.summary.FileWriter('log/lstm_forecast_val', flush_secs=10)

print("Result with timesteps = " + str(timesteps) + ", predict timesteps = " + str(pred_timesteps) + ", n_hidden = " + str(n_hidden) + "\n")

# Soft placement allows placing on CPU ops without GPU implementation.
session_config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
session_config.gpu_options.per_process_gpu_memory_fraction = 0.8
session_config.gpu_options.visible_device_list = '2,3'

# Start training
with tf.Session(config=session_config) as sess:
    # Run the initializer
    sess.run(init)

    if is_training:
        for step in range(1, training_steps+1):
            # Make the training batch for each step
            batch_x, batch_y, batch_ymap = batch_creator(X_train, batch_size, X_train.shape[0])

            # Run optimization
            _, train_loss, summary = sess.run([train_op, loss, merged], feed_dict = {x: batch_x, y: batch_y, y_map: batch_ymap})
            #train_writer.add_summary(summary, step)

            # compute error on validate set
            batch_x, batch_y, batch_ymap = batch_creator(X_val, batch_size, X_val.shape[0])

            [validate_loss, summary] = sess.run([loss, merged], feed_dict={x: batch_x, y: batch_y, y_map: batch_ymap})
            #val_writer.add_summary(summary, step)

            # Print result
            if step % display_step == 0 or step == 1:
                print("Iter = " + str(step) + ". Train error = {:.6f}. Validate error = {:.6f}".format(train_loss, validate_loss))
                    
        print("Training Finished!")
        # Save model weights to disk
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

    # Test error
    saver.restore(sess, model_path)
    loss_test = 0
    elapsed_time = 0
    batch_size = 1
    test_steps = int(X_test.shape[0] / batch_size)
    for i in range(test_steps):
        batch_x, batch_y, batch_ymap = batch_creator(X_test, batch_size, X_test.shape[0])
        
        start_time = time.time()
        [out] = sess.run([output], feed_dict={x: batch_x, y: batch_y, y_map: batch_ymap})

        inv_out = scaler.inverse_transform(out.flatten().reshape(-1, 1))
        inv_yhat = np.multiply(inv_out, batch_ymap.flatten().reshape(-1, 1))
        inv_y = scaler.inverse_transform(batch_y.flatten().reshape(-1, 1))
        loss_value = sqrt(mean_squared_error(inv_y, inv_yhat) * loss_ratio)

        if i % 600 == 0:
            print(loss_value)
            '''fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.set_title('prediction')
            X_i = inv_out.reshape(32, 32)
            ax.imshow(X_i, cmap='gray', interpolation='none')

            ax = fig.add_subplot(122)
            ax.set_title('actual')
            X_i1 = inv_y.reshape(32, 32)
            ax.imshow(X_i1, cmap='gray', interpolation='none')

            plt.show()'''

        elapsed_time += time.time() - start_time
        loss_test += loss_value

    # Print validate error
    print("Test Error = {:.6f}. Elapsed time = {:.3f}".format(loss_test/test_steps, elapsed_time/test_steps))

    output_path = "output/lstm_forecast.txt"
    outfile = open(output_path, 'a')
    outfile.write('\n')
    outfile.write("Result with timesteps = " + str(timesteps) + ", predict timesteps = " + str(pred_timesteps) + ", n_hidden = " + str(n_hidden) + "\n")
    outfile.write("Test Error = {:.6f}. Elapsed time = {:.3f}\n".format(loss_test/test_steps, elapsed_time/test_steps))


