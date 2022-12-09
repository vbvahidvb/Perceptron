import numpy as np
import matplotlib.pyplot as plt

def max_like(x , cov, mu):
    f = (1/((2*np.pi)*(np.linalg.det(cov)**0.5))) * np.exp(-0.5 * np.matmul(np.matmul(np.transpose(x-mu),np.linalg.inv(cov)),(x-mu)))
    return f


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

test_class1 = np.append(class_1[:250,:],label_class1, axis=1);    "Create test class 1 with labels"
test_class2 = np.append(class_2[:250,:],label_class2, axis=1);    "Create test class 2 with labels"
test_set = np.append(test_class1, test_class2, axis=0);    "Create test set"
np.random.shuffle(test_set);    "Shuffle test set data"

X_test = np.array(test_set[:,:2]);    "Split input test data"
y_test = np.array(test_set[:,2].reshape([500,1]));    "Split output train data"

#Plot data
x = test_set[:, [0, 1]]
y = test_set[:, -1].astype(int)
plt.figure(figsize=(5,5))
cl2 = plt.scatter(x[:,0][y==0], x[:,1][y==0], s=3, c='r')
cl1 = plt.scatter(x[:,0][y==1], x[:,1][y==1], s=3, c='b')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Test set scatter plot')
plt.legend((cl1, cl2), ('Class 1', 'Class 2'))
plt.savefig('Data_Q7.png')
plt.show()

c1_cor=0
c2_cor=0
c1_incor=0
c2_incor = 0

for i in test_set:
    prob_c1 = max_like(i[0:1], cov1, mu1)
    prob_c2 = max_like(i[0:1], cov2, mu2)

    if prob_c1>prob_c2 and i[2] == 1:
        c1_cor += 1
    elif prob_c2>prob_c1 and i[2] == 0:
        c2_cor += 1
    elif prob_c1>prob_c2 and i[2] ==0:
        c1_incor += 1
    elif prob_c2>prob_c1 and i[2] ==1:
        c2_incor += 1

print('\nMaximum Likelihood result:')
print('Correct C1: ' + str(c1_cor) + '\nIncorrect C1: ' + str(c1_incor) + '\nCorrect C2: ' + str(c2_cor) + '\nIncorrect C2: ' + str(c2_incor))
print('Accuracy: ' + str((c1_cor + c2_cor)/len(test_set)))
