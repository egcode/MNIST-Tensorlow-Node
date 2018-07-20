import numpy as np
import tensorflow as tf

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 6x6x1=>6 stride 1        W1 [5, 5, 1, 6]        B1 [6]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x6=>12 stride 2       W2 [5, 5, 6, 12]        B2 [12]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer 4x4x12=>24 stride 2      W3 [4, 4, 12, 24]       B3 [24]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout) W4 [7*7*24, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 10]


# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has 10 softmax neurons)
K = 6  # first convolutional layer output depth
L = 12  # second convolutional layer output depth
M = 24  # third convolutional layer
N = 200  # fully connected layer


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    lr -- variable learning rate
    pkeep -- Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
    step -- step for variable learning rate
    """

    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))
    
    lr = tf.placeholder(tf.float32)
    pkeep = tf.placeholder(tf.float32)
    step = tf.placeholder(tf.int32)

    return X, Y, lr, pkeep, step

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. 
    
    Returns:
    parameters -- a dictionary of tensors 
    """
        
    W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
    
    W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
    W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
    B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

    params = {"W1": W1,
              "B1": B1,
              "W2": W2,
              "B2": B2,
              "W3": W3,
              "B3": B3,
              "W4": W4,
              "B4": B4,
              "W5": W5,
              "B5": B5}
    
    return params


def forward_propagation(X, parameters, pkeep):
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    B1 = parameters['B1']
    W2 = parameters['W2']
    B2 = parameters['B2']
    W3 = parameters['W3']
    B3 = parameters['B3']

    W4 = parameters['W4']
    B4 = parameters['B4']
    W5 = parameters['W5']
    B5 = parameters['B5']

    X = tf.cast(X, tf.float32)

    # The model
    stride = 1  # output is 28x28
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2  # output is 14x14
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    stride = 2  # output is 7x7
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
    
    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])
    
    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    YY4 = tf.nn.dropout(Y4, pkeep)
    Y_hat_logits = tf.matmul(YY4, W5) + B5
    Y_hat = tf.nn.softmax(Y_hat_logits)

    return Y_hat, Y_hat_logits

def compute_cost(Y_hat_logits, Y):
    """
    Computes the cost
    
    Arguments:
    Y_hat_logits -- output of forward propagation 
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    # cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 64  images
    # TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
    # problems with log(0) which is NaN
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y_hat_logits, labels=Y)
    cross_entropy = tf.reduce_mean(cross_entropy)*100
        
    return cross_entropy



#####################################
    
def forward_propagation_with_layers(X, parameters, pkeep):
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    B1 = parameters['B1']
    W2 = parameters['W2']
    B2 = parameters['B2']
    W3 = parameters['W3']
    B3 = parameters['B3']

    W4 = parameters['W4']
    B4 = parameters['B4']
    W5 = parameters['W5']
    B5 = parameters['B5']

    X = tf.cast(X, tf.float32)

    # The model
    stride = 1  # output is 28x28
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2  # output is 14x14
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    stride = 2  # output is 7x7
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)
    
    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])
    
    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    YY4 = tf.nn.dropout(Y4, pkeep)
    Y_hat_logits = tf.matmul(YY4, W5) + B5
    Y_hat = tf.nn.softmax(Y_hat_logits)

    return Y_hat, Y_hat_logits, Y1, Y2, Y3, Y4
