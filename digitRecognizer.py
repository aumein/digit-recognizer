import numpy as np
import pandas as pd
from matplotlib import pyplot as pp
#HAS NOT BEEN COMPLETED

data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

data.head()

data = np.array(data)
row, col = data.shape
np.random.shuffle(data)

dev_data = data[0:1000].T #DEV SET
dev_Y = dev_data[0] #labels on each example
dev_X =dev_data[1:row] #example data in DEV SET

train_data = data[1000:row].T #TRAINING SET
train_Y = train_data[0] #labels
train_X = train_data[1:row] #example data in TRAINING SET

def init_params():
    W1 = np.random.randn(10,784)
    W2 = np.random.randn(10,10)
    b1 = np.random.randn(10, 1)
    b2 = np.random.randn(10, 1)
    
    #print vars
    #print('w1 = ', W1)
    #print("w2 = ", W2)
    #print("b1 = ", b1)
    #print("b2 = ", b2)
    return W1, W2, b1, b2

def activation(z):
    #ReLU
    return np.maximum(0,z)

def deriv_activation(z):
    #ReLU trick
    return z > 0

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

def one_hot(Y):
    one_hot = np.zeros(Y.size, Y.max()+1)
    one_hot[np.arange(Y.size),Y] = 1
    one_hot = one_hot.T
    return one_hot

def forward_prop(W1, W2, b1, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = activation(Z1)
    
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    
        #print vars
    #print('w1 = ', W1)
    #print("w2 = ", W2)
    #print("b1 = ", b1)
    #print("b2 = ", b2)
                       
    return Z1, A1, Z2, A2
    
def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y): #BROKEN db1 and db2 and dw1 and dw2 return nan arrays
    #print("x = ", X)
    #print("y = ", Y)
    
    m = Y.size
    Y = one_hot(Y)
    dz2 = A2 - Y
    dw2 = 1/m * dz2.dot(A1.T)
    db2 = 1/m * np.sum(dz2, 0) #CHECK AFTER
    
    dz1 = W2.T.dot(dz2) * deriv_activation(Z1)
    dw1 = 1/m* dz1.dot(X.T)
    db1 = 1/m * np.sum(dz1, 0) #CHECKKK
    
        #print vars
    #print('w1 = ', W1)
    #print("w2 = ", W2)
    #print("b1 = ", b1)
    #print("b2 = ", b2)
    
    
    return dw1, db1, dw2, db2
                       
def update_params(W1, W2, b1, b2, dW1, dW2, db1, db2, alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    
        #print vars
    #print('w1 = ', W1)
    #print("w2 = ", W2)
    #print("b1 = ", b1)
    #print("b2 = ", b2)

    return W1, b1, W2, b2

def predictions(A2):
    return np.argmax(A2, 0)

def accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

    

def evolution(X, Y, generations, alpha):
    W1, W2, b1, b2 = init_params()
    for i in range(generations):
        Z1, A1, Z2, A2 = forward_prop(W1, W2, b1, b2, train_X)
        dw1, db1, dw2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, train_X, train_Y)
        W1, b1, W2, b2 = update_params(W1, W2, b1, b2, dw1, dw2, db1, db2, alpha)
        
        if(i % 5 == 0):
            print("Generation: ", i)
            print("Accuracy: ", accuracy(predictions(A2), Y))
            
    return W1, b1, W2, b2

W1, b1, W2, b2 = evolution(train_X, train_Y, 1000, 0.5)
