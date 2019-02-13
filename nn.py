import numpy as np
import time

class NN(object):
  
  def __init__(self, hidden_dims=(480,256), mode='train',
               datapath='data/mnist.pkl.npy', model_path=None,
               weight_init='glorot', activation_type='relu',
               step_size=0.02, batch_size=32):
    np.random.seed(1)
      
    self.mode = mode
    self.weight_init = weight_init
    self.step_size = step_size
    self.batch_size = batch_size

    # Set an activation function.
    if activation_type == 'relu':
        self.activation = self.relu
        self.inverse_activation = self.relu_derivative
    else:
        self.activation = self.sigmoid
        self.inverse_activation = self.sigmoid_derivative
   
    self.h0 = 784
    self.h1 = hidden_dims[0]
    self.h2 = hidden_dims[1]
    self.h3 = 10
  
    # Load and split the mnist data
    data = np.load(datapath)
    self.X_train = data[0][0]
    self.y_train = self.one_hot(data[0][1])
    self.X_valid = data[1][0]
    self.y_valid = self.one_hot(data[1][1])
    self.X_test = data[2][0]
    self.y_test = self.one_hot(data[2][1])

    # Initialize weights, either randomly or load from model
    self.model_path = model_path
    self.initialize_weights()

  def one_hot(self, y):
    y2 = np.zeros((y.shape[0], self.h3))
    y2[np.arange(y.shape[0]), y] = 1
    return y2
  
  def initialize_weights(self):
    self.W1 = self.weight_matrix((self.h1, self.h0))
    self.b1 = np.zeros((self.h1, 1))

    self.W2 = self.weight_matrix((self.h2, self.h1))
    self.b2 = np.zeros((self.h2, 1))

    self.W3 = self.weight_matrix((self.h3, self.h2))
    self.b3 = np.zeros((self.h3, 1))
      
  def weight_matrix(self, size):
    if self.weight_init == 'zero':
      return np.zeros(size)
    elif self.weight_init == 'normal':
      return np.random.normal(0, 1, size)
    elif self.weight_init == 'glorot':
      dl = np.sqrt(float(6) / (size[0] + size[1]))
      return np.random.uniform(-dl, dl, size)
  
  def forward(self, input, label):
    self.a0 = input
    self.y = label

    self.z1 = np.dot(self.W1, self.a0) + self.b1
    self.a1 = self.activation(self.z1)

    self.z2 = np.dot(self.W2, self.a1) + self.b2
    self.a2 = self.activation(self.z2)

    self.z3 = np.dot(self.W3, self.a2) + self.b3
    self.a3 = self.softmax(self.z3)

    self.L = self.ce_loss(self.a3, self.y)
    
  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def sigmoid_derivative(self, a):
    s = 1 / (1 + np.exp(-a))
    return s * (1 - s)

  def relu(self, z):
    return np.maximum(0, z)
  
  def relu_derivative(self, z):
    z[z < 0] = 0
    return z 
    
  def ce_loss(self, a, y):
    return np.sum(-np.log(np.sum(a * y, axis=0))) / y.shape[1]

  def softmax(self, z):
    s = np.exp(z- np.amax(z, axis=0))
    return s / np.sum(s, axis=0)
  
  def inverse_softmax_ce_loss(self, a, y):
    return (a - y) / y.shape[1]
    
  def backward(self):
    # loss with respect to outputs
    dz3 = self.inverse_softmax_ce_loss(self.a3, self.y)
    self.dW3 = np.dot(dz3, self.a2.T)
    self.db3 = np.sum(dz3)
    
    da2 = np.dot(self.W3.T, dz3)
    dz2 = np.multiply(da2, self.inverse_activation(self.z2))
    self.dW2 = np.dot(dz2, self.a1.T)
    self.db2 = np.sum(dz2)
    
    da1 = np.dot(self.W2.T, dz2)
    dz1 = np.multiply(da1, self.inverse_activation(self.z1))
    self.dW1 = np.dot(dz1, self.a0.T)
    self.db1 = np.sum(dz1)
    
  def update(self):
    self.W3 -= self.step_size * self.dW3
    self.b3 -= self.step_size * self.db3
    self.W2 -= self.step_size * self.dW2
    self.b2 -= self.step_size * self.db2
    self.W1 -= self.step_size * self.dW1
    self.b1 -= self.step_size * self.db1

  def train(self):
    X = self.X_train
    y = self.y_train
    n_train = X.shape[0]
    for ep in range(10):
      start_time = time.time()
      p = np.random.permutation(n_train)
      X, y = X[p], y[p]
      total_loss = 0
      for i in range(0, n_train - self.batch_size, self.batch_size):
        input = X[i:i + self.batch_size].T.reshape(self.h0, self.batch_size)
        label = y[i:i + self.batch_size].T.reshape(self.h3, self.batch_size)
        self.forward(input, label)
        self.backward()
        self.update()
        total_loss += np.sum(self.L)
      print(total_loss / n_train)
      print("--- %s seconds ---" % (time.time() - start_time))
      self.test(self.X_valid, self.y_valid)
    
  def test(self, X, y):
    input = X.T
    label = y.T
    self.forward(input, label)
    est = np.argmax(self.a3, axis=0)
    gt = np.argmax(label, axis=0)
    n_correct = X.shape[0] - np.count_nonzero(est - gt)
    print(float(n_correct) / X.shape[0])
