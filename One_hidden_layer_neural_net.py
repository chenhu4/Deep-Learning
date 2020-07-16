import numpy as np
import h5py
import time
import copy
from random import randint
import matplotlib.pyplot as plt
#load MNIST data
MNIST_data = h5py.File('C:\\Users\\76754\\Desktop\\deep_learning\\MNISTdata_1.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )# 60000 784
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0]))
MNIST_data.close()
####################################################################################
#Implementation of stochastic gradient descent algorithm
#number of inputs
num_inputs = 28*28
#number of outputs
num_outputs = 10
#number of hidden layer units
num_layer = 128
num_samples=x_train.shape[0]

#model_grads = copy.deepcopy(model)


def softmax_function(z):
    '''
    input: z    (num_outputs,1)
    output: ZZ  (num_outputs,1)
    the sum of the ZZ equals to 1. Transferring the output into a probability distribution.
    '''
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

def Relu(z):
    '''
    input: z (num_layer,num_inputs)
    output: zz(num_layer,num_inputs)
    '''
    zz=np.maximum(0,z)
    return zz

def forward(x, model):
    '''
    input: x (num_inputs,1) model
    output:y(num_outputs,1),z(num_layer,1),h(num_layer,1)
    '''

    z = np.dot(model['W'], x.T)+model['b1']
    h =np.maximum(0,z) #Rrlu
    f=np.dot(model['C'], h)+model['b2']
    y=softmax_function(f)
    return y,z,h

def cost_function(y,y_train):

    cost = -np.sum(np.log(y[y_train]))

    return cost


def backward(x,index,index_sample,h,y):
    onehot = np.zeros((num_outputs, 1))
    onehot[index] = 1
    du = y.reshape(10, 1) - onehot
    h_index = h.reshape(128, 1)
    drelu=np.ones([128,1])
    drelu[h_index<=0]=0
    dsigma=np.dot(model['C'].T,du)
    dc=np.dot(du, h.reshape(128,1).T)
    db2=du
    db1=dsigma*drelu
    dw=np.dot(db1,x[index_sample,:].reshape(1,784))
    model_grads['dw']=dw
    model_grads['db1'] = db1
    model_grads['db2'] = db2
    model_grads['dc'] = dc
    return model_grads


model = {}
model['W'] = np.random.normal(0,1/num_inputs,(num_layer,num_inputs) )
model['C'] = np.random.normal(0,1/num_layer,(num_outputs,num_layer))
model['b1'] = np.random.normal(0,1,(num_layer,1))
model['b2'] = np.random.normal(0,1,(num_outputs,1))
model_grads = copy.deepcopy(model)

LR=0.01
i=0
a=0
num_epochs = 10
Accuracy=[]
cost=[]
output=[]
time1 = time.time()
cost_mean=[]
for num in range(num_epochs):
    if (num>=4):
        LR=0.001
    if (num>=6):
        LR=0.0001
    for i in range(40000):
        index_sample=randint(0,x_train.shape[0]-1)
        index=y_train[index_sample]
        y,z,h=forward(x_train[index_sample,:].reshape(784,1).T, model)

        model_grads=backward(x_train,index,index_sample,h,y)
        model['W'] =model['W']-LR*model_grads['dw']
        model['C'] =model['C'] - LR * model_grads['dc']
        model['b1']=model['b1'] - LR * model_grads['db1']
        model['b2']=model['b2'] - LR * model_grads['db2']

    for i in range(60000):
        y,z,h=forward(x_train[i,:].reshape(784,1).T, model)
        cost.append(cost_function(y,y_train[i]))
    cost_mean.append(np.mean(cost))

    correct = 0

    for i in range(10000):

        y,z,h=forward(x_test[i,:].reshape(784,1).T, model)
        prediction=np.argmax(y)
        if(prediction==y_test[i]):
            correct+=1


    accuracy=correct/np.float(10000)
    Accuracy.append(accuracy)
    print(accuracy*100,'%')
    #a+=1
plt.figure(1)
plt.plot(range(num_epochs),cost_mean)
plt.xlabel('Iteration')
plt.ylabel('Cross-entropy error')
plt.title('Iteration vs Cross-entropy error')

plt.figure(2)
plt.plot(range(num_epochs),Accuracy)
plt.xlabel('Iteration')
plt.ylabel('Test Accuracy')
plt.title('Iteration vs Test Accuracy')
time2 = time.time()
print('Running time is',time2-time1,'s')

