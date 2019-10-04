#Import the necassary libraries and files
import numpy as np
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import pandas as pd
from load_and_explore_data import X_train, y_train, X_valid, y_valid, X_test, y_test

class LeNet:
    
    def __init__(self, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):
        #Hyperparameters
        self.mu = mu
        self.sigma = sigma

        #Convolutional Layer 1: Input = 32x32x3. Output = 28x28x6.
        self.filter1_width = 5
        self.filter1_height = 5
        self.input1_channels = 3
        self.conv1_output = 6
        #Weights and bias
        self.conv1_weight = tf.Variable(tf.truncated_normal(
            shape=(self.filter1_width, self.filter1_height, self.input1_channels, self.conv1_output),
            mean = self.mu, stddev = self.sigma))
        self.conv1_bias = tf.Variable(tf.zeros(self.conv1_output))
        #Apply Convolution
        self.conv1 = tf.nn.conv2d(x, self.conv1_weight, strides=[1, 1, 1, 1], padding='VALID') + self.conv1_bias
        
        #Activation
        self.conv1 = tf.nn.relu(self.conv1)
        
        #Pooling: Input = 28x28x6. Output = 14x14x6.
        self.conv1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        #Convolutional Layer 2: Input = 14x14x6. Output = 10x10x16.
        self.filter2_width = 5
        self.filter2_height = 5
        self.input2_channels = 6
        self.conv2_output = 16
        #Weight and bias
        self.conv2_weight = tf.Variable(tf.truncated_normal(
            shape=(self.filter2_width, self.filter2_height, self.input2_channels, self.conv2_output),
            mean = self.mu, stddev = self.sigma))
        self.conv2_bias = tf.Variable(tf.zeros(self.conv2_output))
        #Apply Convolution
        self.conv2 = tf.nn.conv2d(self.conv1, self.conv2_weight, strides=[1, 1, 1, 1], padding='VALID') + self.conv2_bias
        
        #Activation:
        self.conv2 = tf.nn.relu(self.conv2)
        
        #Pooling: Input = 10x10x16. Output = 5x5x16.
        self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        
        #Flattening: Input = 5x5x16. Output = 400.
        self.fully_connected0 = flatten(self.conv2)
        
        #Fully Connected Layer 1: Input = 400. Output = 120.
        self.connected1_weights = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = self.mu, stddev = self.sigma))
        self.connected1_bias = tf.Variable(tf.zeros(120))
        self.fully_connected1 = tf.add((tf.matmul(self.fully_connected0, self.connected1_weights)), self.connected1_bias)
        
        #Activation:
        self.fully_connected1 = tf.nn.relu(self.fully_connected1)
        #Dropout
        self.fully_connected1 = tf.nn.dropout(self.fully_connected1, keep_prob)
    
        #Fully Connected Layer 2: Input = 120. Output = 84.
        self.connected2_weights = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = self.mu, stddev = self.sigma))
        self.connected2_bias = tf.Variable(tf.zeros(84))
        self.fully_connected2 = tf.add((tf.matmul(self.fully_connected1, self.connected2_weights)), self.connected2_bias)
        
        #Activation.
        self.fully_connected2 = tf.nn.relu(self.fully_connected2)
        #Dropout
        self.fully_connected2 = tf.nn.dropout(self.fully_connected2, keep_prob)
    
        #Fully Connected Layer 5: Input = 84. Output = 43.
        self.output_weights = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = self.mu, stddev = self.sigma))
        self.output_bias = tf.Variable(tf.zeros(43))
        self.logits =  tf.add((tf.matmul(self.fully_connected2, self.output_weights)), self.output_bias)

        #Training operation
        self.one_hot_y = tf.one_hot(y, n_out)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_y)
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        #Accuracy operation
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        #Saving all variables
        self.saver = tf.train.Saver()
    
    def y_predict(self, X_data, batch_size=64):
        num_examples = len(X_data)
        y_pred = np.zeros(num_examples, dtype=np.int32)
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x = X_data[offset:offset+batch_size]
            y_pred[offset:offset+batch_size] = sess.run(tf.argmax(self.logits, 1), 
                               feed_dict={x:batch_x, keep_prob:1, keep_prob_conv:1})
        return y_pred
    
    def evaluate(self, X_data, y_data, batch_size=64):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
            accuracy = sess.run(self.accuracy_operation, 
                                feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0 })
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
    
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))

keep_prob = tf.placeholder(tf.float32)
keep_prob_conv = tf.placeholder(tf.float32)

epochs = 50
batch_size = 200
DIR = 'Saved_Models'

LeNet_Model = LeNet(n_out=43)
model_name = "LeNet"

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(y_train)
    print("Training...")
    print()
    for i in range(epochs):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(LeNet_Model.training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob : 0.4, keep_prob_conv: 0.7})
        #Accuracy on validation set
        validation_accuracy = LeNet_Model.evaluate(X_valid, y_valid)
        print("EPOCH {}: Validation Accuracy = {:.3f}%".format(i+1, (validation_accuracy*100)))
    #Save the model
    LeNet_Model.saver.save(sess, os.path.join(DIR, model_name))
    print("Model saved")
    #Accuracy on test set
    test_accuracy = LeNet_Model.evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy*100))
    #Alternative way for calculating the accuracy on test set
    y_pred = LeNet_Model.y_predict(X_test)
    #acc = np.sum(y_pred==y_test)/np.size(y_pred)
    #print("Test Accuracy based on y_pred = {:.3f}%".format(acc*100))
    
with tf.Session() as sess:
    #Restore LeNet model from disk
    LeNet_Model.saver.restore(sess, r'C:\Users\Bill\Desktop\traffic_signs_restricted_area\Saved_Models\LeNet')
    print("LeNet model restored")
    #Print the test accuracy
    test_accuracy = LeNet_Model.evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy*100))

#Write into a CSV file with columns ImageId & Label
submission = pd.DataFrame({"ImageId": list(range(0, len(y_pred))), "Predicted Label": y_pred, "True Label": y_test})
submission.to_csv("LeNet_Results.csv", index=False)