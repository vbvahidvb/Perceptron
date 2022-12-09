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

def fit(X, y, param, iterations, lr):
    "Multi-layer perceptron trained with backpropagation"

    #storage errors after each iteration
    errors = []
    accuracy_model = []

    for i in range(iterations):
        
        #Forward-propagation
        Z1 = linear_function(param['W1'], X, param['b1'])
        S1 = sigmoid(Z1)
        Z2 = linear_function(param['W2'], S1, param['b2'])
        S2 = sigmoid(Z2)
        Z3 = linear_function(param['W3'], S2, param['b3'])
        S3 = sigmoid(Z3)
        
        #Error computation
        error = cost_function(S3, y)
        errors.append(error)
        
        #Accuracy computation
        predict_test = np.where(S3 >= 0.5, 1, 0)
        num_correct_predictions = (predict_test == y).sum()
        accuracy_model.append((num_correct_predictions / y.shape[0]) * 100)

        #Backpropagation
        #update output weights
        delta3 = (S3 - y)* S3*(1-S3)
        W3_gradients = S2.T @ delta3
        param["W3"] = param["W3"] - W3_gradients * lr

        #update output bias
        param["b3"] = param["b3"] - np.sum(delta3, axis=0, keepdims=True) * lr

        #update hidden weights
        delta2 = (delta3 @ param["W3"].T )* S2*(1-S2)
        W2_gradients = S1.T @ delta2 
        param["W2"] = param["W2"] - W2_gradients * lr

        #update hidden bias
        param["b2"] = param["b2"] - np.sum(delta2, axis=0, keepdims=True) * lr

        delta1 = (delta2 @ param["W2"].T )* S1*(1-S1)
        W1_gradients = X.T @ delta1 
        param["W1"] = param["W1"] - W1_gradients * lr

        #update hidden bias
        param["b1"] = param["b1"] - np.sum(delta1, axis=0, keepdims=True) * lr
        
    return errors, param, accuracy_model

def predict(X, W1, W2, W3, b1, b2, b3):
    """computes predictions with learned parameters
       First, calculate output of each layer,
       Then, pass them through sigmoid activation function,
       Finally, feed them to the next layer
    """
    
    Z1 = linear_function(W1, X, b1)
    S1 = sigmoid(Z1)
    Z2 = linear_function(W2, S1, b2)
    S2 = sigmoid(Z2)
    Z3 = linear_function(W3, S2, b3)
    S3 = sigmoid(Z3)
    return np.where(S3 >= 0.5, 1, 0)

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
y_train = np.array(train_set[:,2].reshape([500,1])).astype(int);    "Split output train data"

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
plt.savefig('Data_Q5.png')
plt.show()

#Define network parameters
np.random.seed(100) # for reproducibility
W1 = np.random.uniform(size=(2,10)); "n_features: 2, n_neuron_hidden_layer: 10"
b1 = np.random.uniform(size=(1,10)); "n_neuron_hidden_layer: 10"
W2 = np.random.uniform(size=(10,10)); "n_next: 10, n_neuron_hidden_layer: 10"
b2 = np.random.uniform(size=(1,10)); "n_next: 10"
W3 = np.random.uniform(size=(10,1)); "n_output: 1, n_neuron_hidden_layer: 10"
b3 = np.random.uniform(size=(1,1)); "n_output: 1"

param = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3};   'Enter parameters to a dictionary'

#Train train set
errors, param, accuracy_model = fit(X_train, y_train, param, iterations=2000, lr=0.005)

#Predict test set
y_pred = predict(X_test, param["W1"], param["W2"], param["W3"], param["b1"], param["b2"], param["b3"])
num_correct_predictions = (y_pred == y_test).sum()

#Calculate accuracy
accuracy = (num_correct_predictions / y_test.shape[0]) * 100
print('Multi-layer perceptron accuracy with 2 hidden layers: ' + str(format(accuracy, '.2f')))

plt.figure(figsize=(3.15,3.15))
plt.plot(accuracy_model)
plt.xlabel('Iterations')
plt.ylabel('Accuracy(%)')
plt.title('Model with two hidden layers and ten neurons in each layer')
plt.show()