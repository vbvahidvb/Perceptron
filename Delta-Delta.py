import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  "Compute sigmoid values for each sets of scores in x"
  return 1 / (1 + np.exp(-x))

def linear_function(W, X, b):
  "computes net input as dot product"
  return (X @ W)+b

def cost_function(A, y):
  """computes squared error
    A: Output values
    y: Desired values
    """
  return (np.mean(np.power(A - y,2)))/2

def lr_adaptation(lr, now, prev, lambda_coeff=0.001):
    "Apdate learning rate according to current and previous weigths"
    return (lr + lambda_coeff * (now * prev))

def fit(X, y, param, iterations, lr):
    "Multi-layer perceptron trained with backpropagation"

    #storage errors and accuracy after each iteration
    errors = []
    accuracy_model = []
    b2_prev, b1_prev = 0, 0
    W2_prev, W1_prev = np.zeros((10,1)), np.zeros((2,10))
    lr_b1, lr_b2 = lr, lr
    lr_w2, lr_w1 = lr*np.ones((10,1)), lr*np.ones((2,10))

    for i in range(iterations):
        
        #Forward-propagation
        Z1 = linear_function(param['W1'], X, param['b1'])
        S1 = sigmoid(Z1)
        Z2 = linear_function(param['W2'], S1, param['b2'])
        S2 = sigmoid(Z2)
        
        #Error computation
        error = cost_function(S2, y)
        errors.append(error)

        #Accuracy computation
        predict_test = np.where(S2 >= 0.5, 1, 0)
        num_correct_predictions = (predict_test == y).sum()
        accuracy_model.append((num_correct_predictions / y.shape[0]) * 100)

        #Backpropagation
        #update output weights
        delta2 = (S2 - y)* S2*(1-S2)
        W2_gradients = S1.T @ delta2
        lr_w2 = lr_adaptation(lr_w2, W2_gradients, W2_prev)
        param["W2"] = param["W2"] - lr_w2 * W2_gradients
        W2_prev = W2_gradients

        #update output bias
        b2_now = np.sum(delta2, axis=0, keepdims=True)
        lr_b2 = lr_adaptation(lr_b2, b2_now, b2_prev)
        param["b2"] = param["b2"] - b2_now * lr
        b2_prev = b2_now

        #update hidden weights
        delta1 = (delta2 @ param["W2"].T )* S1*(1-S1)
        W1_gradients = X.T @ delta1
        lr_w1 = lr_adaptation(lr_w1, W1_gradients, W1_prev)
        param["W1"] = param["W1"] - lr_w1 * W1_gradients
        W1_prev = W1_gradients

        #update hidden bias
        b1_now = np.sum(delta1, axis=0, keepdims=True)
        lr_b1 = lr_adaptation(lr_b1, b1_now, b1_prev)
        param["b1"] = param["b1"] - b1_now * lr_b1
        b1_prev = b1_now
        
    return errors, param, accuracy_model

def predict(X, W1, W2, b1, b2):
    """computes predictions with learned parameters
       First, calculate output of each layer,
       Then, pass them through sigmoid activation function,
       Finally, feed them to the next layer
    """
    
    Z1 = linear_function(W1, X, b1)
    S1 = sigmoid(Z1)
    Z2 = linear_function(W2, S1, b2)
    S2 = sigmoid(Z2)
    return np.where(S2 >= 0.5, 1, 0)

#Generate Data
class_1 = np.random.normal(loc=(0,0), scale=1, size=(500,2))
mu1 = np.mean(class_1,axis=0)
var1 = np.var(class_1)
cov1 = np.cov(class_1, rowvar=False)
print('Class 1 properties: \n' + '\tMean: ' + str(mu1) + '\n\tVariance: ' + str(var1)  + '\n\tCovariance: ' + str(cov1) + '\n\tShape: ' + str(class_1.shape))

class_2 = np.random.normal(loc=(2,0), scale=2, size=(500,2))
mu2 = np.mean(class_2,axis=0)
var2 = np.var(class_2)
cov2 = np.cov(class_2, rowvar=False)
print('\n\nClass 2 properties: \n' + '\tMean: ' + str(mu2) + '\n\tVariance: ' + str(var2)  + '\n\tCovariance: ' + str(cov2)  + '\n\tShape: ' + str(class_2.shape))

#Creat, shuffle and split data
label_class1 = np.ones((250,1));    "lable class 1 = 1"
label_class2 = np.zeros((250,1));    "lable class 2 = 0"

train_class1 = np.append(class_1[250:,:],label_class1, axis=1);    "Create train class 1 with labels"
train_class2 = np.append(class_2[250:,:],label_class2, axis=1);    "Create train class 2 with labels"
train_set = np.append(train_class1, train_class2, axis=0);    "Create train set"

test_class1 = np.append(class_1[:250,:],label_class1, axis=1);    "Create test class 1 with labels"
test_class2 = np.append(class_2[:250,:],label_class2, axis=1);    "Create test class 2 with labels"
test_set = np.append(test_class1, test_class2, axis=0);    "Create test set"

np.random.shuffle(train_set);    "Shuffle train set data"
np.random.shuffle(test_set);    "Shuffle test set data"

X_train = np.array(train_set[:,:2]);    "Split input train data"
y_train = np.array(train_set[:,2].reshape([500,1]));    "Split output train data"

X_test = np.array(test_set[:,:2]);    "Split input test data"
y_test = np.array(test_set[:,2].reshape([500,1]));    "Split output train data"

#Plot data
x = train_set[:, [0, 1]]
y = train_set[:, -1].astype(int)
plt.figure(figsize=(5,5))
cl2 = plt.scatter(x[:,0][y==0], x[:,1][y==0], s=3, c='r')
cl1 = plt.scatter(x[:,0][y==1], x[:,1][y==1], s=3, c='b')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Train set scatter plot')
plt.legend((cl1, cl2), ('Class 1', 'Class 2'))
plt.savefig('Data_Q6.png')
plt.show()

Errors = []

np.random.seed(100) # for reproducibility
W1 = np.random.uniform(size=(2,10)); "n_features: 2, n_neuron_hidden_layer: i"
b1 = np.random.uniform(size=(1,10)); "n_neuron_hidden_layer: i"
W2 = np.random.uniform(size=(10,1)); "n_output: 1, n_neuron_hidden_layer: i"
b2 = np.random.uniform(size=(1,1)); "n_output: 1"

param = {"W1": W1, "b1": b1, "W2": W2, "b2": b2};   'Enter parameters to a dictionary'

#Train train set
errors, param, accuracy_fit = fit(X_train, y_train, param, iterations=5000, lr=0.005)
Errors.append(errors)

#Predict test set
y_pred = predict(X_test, param["W1"], param["W2"], param["b1"], param["b2"])
num_correct_predictions = (y_pred == y_test).sum()

#Calculate accuracy
accuracy = (num_correct_predictions / y_test.shape[0]) * 100
print('Multi-layer perceptron accuracy with Delta-Delta learning rate adaptation: ' + str(format(accuracy, '.2f')))


plt.plot(accuracy_fit)
plt.xlabel('Iterations')
plt.ylabel('Accuracy(%)')
plt.title('Model with Delta-Delta learning rate adaptation algorithm')
plt.show()
