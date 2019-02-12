import numpy as np

class NN(object):
  
  def __init__(self, hidden_dims=(50,50), n_hidden=2, mode='train', datapath='data/mnist.pkl.npy', model_path=None):
    self.mode = mode
    
    # Load and split the mnist data
    data = np.load(datapath)
    self.X_train = data[0][0][0:10000]
    self.y_train = data[0][1][0:10000]
    self.X_valid = data[1][0]
    self.y_valid = data[1][1]
    self.X_test = data[2][0]
    self.y_test = data[2][1]
    
    self.input_dim = self.X_train.shape[1]
    self.output_dim = 10
    
    # Initialize weights, either randomly or load from model
    self.model_path = model_path
    self.n_hidden = n_hidden
    self.initialize_weights(hidden_dims)
      
  def initialize_weights(self, hidden_dims):
    dims = [self.input_dim] + [x for x in hidden_dims] + [self.output_dim]
    self.params = {}
    for i in range(self.n_hidden + 1):
      self.params['W_' + str(i + 1)] = np.random.normal(0, 1, (dims[i + 1], dims[i]))
      self.params['b_' + str(i + 1)] = np.zeros((dims[i + 1], 1))
  
  def forward(self, input, label):
    self.cache = {}
    a = input
    for i in range(self.n_hidden + 1):
      # Do a forward step a = activation(W*a_prev + b)
      a_prev = a
      z = np.dot(self.params['W_' + str(i + 1)], a_prev) + self.params['b_' + str(i + 1)]
      if i < self.n_hidden:
        a = self.activation(z)
      else:
        # In the output layer, the activation is always softmax.
        a = self.softmax(z)
      self.cache['z_' + str(i + 1)] = z
      self.cache['a_' + str(i + 1)] = a
    self.loss = self.ce_loss(a, label)
    
  def activation(self, z):
    return 1 / (1 + np.exp(-z))
  
  def inverse_activation(self, a):
    s = 1 / (1 + np.exp(-a))
    return s * (1 - s)
    
  def ce_loss(self, predictions, label):
    return -np.log(predictions)[label]
    
  def softmax(self, z):
    s = np.exp(z)
    return s / np.sum(s)
  
  def inverse_softmax_ce_loss(self, input, label):
    s = self.softmax(input)
    s[label] -= 1
    return s
    
  def backward(self, input, label):
    self.grads = {}
    
    # loss with respect to outputs
    dz3 = self.inverse_softmax_ce_loss(self.cache['z_3'], label)
    dw3 = np.dot(dz3, self.cache['a_2'].T)
    db3 = np.copy(dz3)
    
    da2 = np.dot(self.params['W_3'].T, dz3)
    dz2 = np.multiply(da2, self.inverse_activation(self.cache['z_2']))
    dw2 = np.dot(dz2, self.cache['a_1'].T)
    db2 = np.copy(dz2)
    
    da1 = np.dot(self.params['W_2'].T, dz2)
    dz1 = np.multiply(da1, self.inverse_activation(self.cache['z_1']))
    dw1 = np.dot(dz1, input.T)
    db1 = np.copy(dz1)
    
    self.grads['W_3'] = dw3
    self.grads['W_2'] = dw2
    self.grads['W_1'] = dw1
    
    self.grads['b_3'] = db3
    self.grads['b_2'] = db2
    self.grads['b_1'] = db1
    
  def update(self):
    step_size = 0.001
    for i in range(self.n_hidden + 1):
      self.params['W_' + str(i + 1)] = self.params['W_' + str(i + 1)] - step_size * self.grads['W_' + str(i + 1)]
      self.params['b_' + str(i + 1)] = self.params['b_' + str(i + 1)] - step_size * self.grads['b_' + str(i + 1)]
    
  def train(self):
    n_train = self.X_train.shape[0]
    for ep in range(1000):
      total_loss = 0
      for i in range(n_train):
        input = self.X_train[i].reshape(self.input_dim, 1)
        label = self.y_train[i]
        self.forward(input, label)
        self.backward(input, label)
        self.update()
        total_loss += self.loss
      print(total_loss / n_train)
      self.test()
    
  def test(self):
    n_train = self.X_train.shape[0]
    n_correct = 0
    for i in range(n_train):
        input = self.X_train[i].reshape(self.input_dim, 1)
        label = self.y_train[i]
        self.forward(input, label)
        guess = np.argmax(self.cache['a_3'])
        if guess == label:
            n_correct += 1
    print(float(n_correct) / n_train)
  
nn = NN()
nn.train()
