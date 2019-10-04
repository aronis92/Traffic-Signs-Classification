# Importing Python libraries
import numpy as np
from sklearn.utils import shuffle
import os
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from load_and_explore_data import X_train, y_train, X_valid, y_valid, X_test, y_test

class VGGnet:
    
    def __init__(self, n_out=43, mu=0, sigma=0.1, learning_rate=0.001):
        #Hyperparameters
        self.mu = mu
        self.sigma = sigma

        #Convolutional Layer 1: Input = 32x32x3. Output = 32x32x32.
        self.filter1_width = 3
        self.filter1_height = 3
        self.input1_channels = 3
        self.conv1_output = 32
        #Weights and bias
        self.conv1_W = tf.Variable(tf.truncated_normal(shape=(self.filter1_width, self.filter1_height, self.input1_channels, self.conv1_output), mean = self.mu, stddev = self.sigma))
        self.conv1_b = tf.Variable(tf.zeros(self.conv1_output))
        #Apply Convolution
        self.conv1   = tf.nn.conv2d(x, self.conv1_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv1_b

        #Activation
        self.conv1 = tf.nn.relu(self.conv1)

        #Convolutional Layer 2: Input = 32x32x32. Output = 32x32x32.
        self.filter2_width = 3
        self.filter2_height = 3
        self.input2_channels = 32
        self.conv2_output = 32
        #Weights and bias
        self.conv2_W = tf.Variable(tf.truncated_normal(shape=(self.filter2_width, self.filter2_height, self.input2_channels, self.conv2_output), mean = self.mu, stddev = self.sigma))
        self.conv2_b = tf.Variable(tf.zeros(self.conv2_output))
        #Apply Convolution
        self.conv2   = tf.nn.conv2d(self.conv1, self.conv2_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_b

        #Activation
        self.conv2 = tf.nn.relu(self.conv2)

        #Pooling: Input = 32x32x32. Output = 16x16x32.
        self.conv2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        #Dropout
        self.conv2 = tf.nn.dropout(self.conv2, keep_prob_conv)

        #Convolutional Layer 3: Input = 16x16x32. Output = 16x16x64.
        self.filter3_width = 3
        self.filter3_height = 3
        self.input3_channels = 32
        self.conv3_output = 64
        #Weights and bias
        self.conv3_W = tf.Variable(tf.truncated_normal(shape=(self.filter3_width, self.filter3_height, self.input3_channels, self.conv3_output), mean = self.mu, stddev = self.sigma))
        self.conv3_b = tf.Variable(tf.zeros(self.conv3_output))
        #Apply Convolution
        self.conv3   = tf.nn.conv2d(self.conv2, self.conv3_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv3_b

        #Activation
        self.conv3 = tf.nn.relu(self.conv3)

        #Convolutional Layer 4: Input = 16x16x64. Output = 16x16x64.
        self.filter4_width = 3
        self.filter4_height = 3
        self.input4_channels = 64
        self.conv4_output = 64
        #Weights and bias
        self.conv4_W = tf.Variable(tf.truncated_normal(shape=(self.filter4_width, self.filter4_height, self.input4_channels, self.conv4_output), mean = self.mu, stddev = self.sigma))
        self.conv4_b = tf.Variable(tf.zeros(self.conv4_output))
        #Apply Convolution
        self.conv4   = tf.nn.conv2d(self.conv3, self.conv4_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv4_b

        #Activation
        self.conv4 = tf.nn.relu(self.conv4)

        #Pooling: Input = 16x16x64. Output = 8x8x64.
        self.conv4 = tf.nn.max_pool(self.conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        #Dropout
        self.conv4 = tf.nn.dropout(self.conv4, keep_prob_conv)

        #Convolutional Layer 5: Input = 8x8x64. Output = 8x8x128.
        self.filter5_width = 3
        self.filter5_height = 3
        self.input5_channels = 64
        self.conv5_output = 128
        #Weights and bias
        self.conv5_W = tf.Variable(tf.truncated_normal(shape=(self.filter5_width, self.filter5_height, self.input5_channels, self.conv5_output), mean = self.mu, stddev = self.sigma))
        self.conv5_b = tf.Variable(tf.zeros(self.conv5_output))
        #Apply Convolution
        self.conv5   = tf.nn.conv2d(self.conv4, self.conv5_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv5_b

        #Activation
        self.conv5 = tf.nn.relu(self.conv5)

        #Convolutional Layer 6: Input = 8x8x128. Output = 8x8x128.
        self.filter6_width = 3
        self.filter6_height = 3
        self.input6_channels = 128
        self.conv6_output = 128
        #Weights and bias
        self.conv6_W = tf.Variable(tf.truncated_normal(shape=(self.filter6_width, self.filter6_height, self.input6_channels, self.conv6_output), mean = self.mu, stddev = self.sigma))
        self.conv6_b = tf.Variable(tf.zeros(self.conv6_output))
        #Apply Convolution
        self.conv6   = tf.nn.conv2d(self.conv5, self.conv6_W, strides=[1, 1, 1, 1], padding='SAME') + self.conv6_b

        #Activation
        self.conv6 = tf.nn.relu(self.conv6)

        #Pooling: Input = 8x8x128. Output = 4x4x128.
        self.conv6 = tf.nn.max_pool(self.conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        #Dropout
        self.conv6 = tf.nn.dropout(self.conv6, keep_prob_conv)

        #Flatten. Input = 4x4x128. Output = 2048.
        self.fc0   = flatten(self.conv6)

        #Fully Connected Layer 7: Input = 2048. Output = 128.
        self.fc1_W = tf.Variable(tf.truncated_normal(shape=(2048, 128), mean = self.mu, stddev = self.sigma))
        self.fc1_b = tf.Variable(tf.zeros(128))
        self.fc1   = tf.matmul(self.fc0, self.fc1_W) + self.fc1_b

        #Activation
        self.fc1    = tf.nn.relu(self.fc1)
        #Dropout
        self.fc1    = tf.nn.dropout(self.fc1, keep_prob)

        #Fully Connected Layer 8: Input = 128. Output = 128.
        self.fc2_W  = tf.Variable(tf.truncated_normal(shape=(128, 128), mean = self.mu, stddev = self.sigma))
        self.fc2_b  = tf.Variable(tf.zeros(128))
        self.fc2    = tf.matmul(self.fc1, self.fc2_W) + self.fc2_b

        #Activation
        self.fc2    = tf.nn.relu(self.fc2)
        #Dropout
        self.fc2    = tf.nn.dropout(self.fc2, keep_prob)

        #Fully Connected Layer 9: Input = 128. Output = n_out.
        self.fc3_W  = tf.Variable(tf.truncated_normal(shape=(128, n_out), mean = self.mu, stddev = self.sigma))
        self.fc3_b  = tf.Variable(tf.zeros(n_out))
        self.logits = tf.matmul(self.fc2, self.fc3_W) + self.fc3_b

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
        
    def y_predict(self, X_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        y_pred = np.zeros(num_examples, dtype=np.int32)
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x = X_data[offset:offset+BATCH_SIZE]
            y_pred[offset:offset+BATCH_SIZE] = sess.run(tf.argmax(self.logits, 1), 
                               feed_dict={x:batch_x, keep_prob:1, keep_prob_conv:1})
        return y_pred
    
    def evaluate(self, X_data, y_data, BATCH_SIZE=64):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(self.accuracy_operation, 
                                feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, keep_prob_conv: 1.0 })
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
    
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))

keep_prob = tf.placeholder(tf.float32)       # For fully-connected layers
keep_prob_conv = tf.placeholder(tf.float32)  # For convolutional layers

EPOCHS = 5
BATCH_SIZE = 200
DIR = 'Saved_Models'

VGGNet_Model = VGGnet(n_out = 43)
model_name = "VGGNet"

one_hot_y_valid = tf.one_hot(y_valid, 43)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(y_train)
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(VGGNet_Model.training_operation, 
            feed_dict={x: batch_x, y: batch_y, keep_prob : 0.5, keep_prob_conv: 0.7})

        validation_accuracy = VGGNet_Model.evaluate(X_valid, y_valid)
        print("EPOCH {}: Validation Accuracy = {:.3f}%".format(i+1, (validation_accuracy*100)))
    VGGNet_Model.saver.save(sess, os.path.join(DIR, model_name))
    print("Model saved")