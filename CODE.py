# Import Some Libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# Import Data 
url = 'C:\\Users\\Karim Aboelazm\\Desktop\\ML\\data.txt'
names = ['X','Y']
data = pd.read_csv(url, header=None, names=names)
# Insert bais 
data.insert(0,'bais',1)
# data contents
cols = data.shape[1]
xval = data.iloc[:,0:cols-1]
yval = data.iloc[:,cols-1:cols]
# Converting contents to matrices
X = np.matrix(xval.values)
Y = np.matrix(yval.values)
W = np.matrix(np.array([0,0]))
# Cost function
def cost_function(x,y,w):
    z = np.sum(np.power((x*w.T)-y,2)/(2*len(x)))
    return z
# Gradient Descent function
def gradient_descent_f(x,y,w,lr,times):
    temp = np.matrix(np.zeros(w.shape))
    prameters = int(w.ravel().shape[1])
    costs = np.zeros(times)
    for i in range(times):
        error = (x*w.T)-y
        for j in range(prameters):
            term = np.multiply(error,x[:,j])
            temp[0,j] = w[0,j] - (lr/len(x))*np.sum(term)
        w = temp
        costs[i] = cost_function(x,y,w)
    return w,costs
# Initial Values 
lr = 0.001
times = 1000
weight,cost = gradient_descent_f(X,Y,W,lr,times)
w = weight
# ploting result 
x = np.linspace(data.X.min(),data.X.max(),100)
f = w[0,0] + x*w[0,1]
fig , ax = plt.subplots(figsize=(5,5))
ax.plot(x,f,'r',label='predictions')
ax.scatter(data.X,data.Y,label='Training data')
ax.legend()
ax.set_xlabel('populations')
ax.set_ylabel('profit')
ax.set_title('prediction profit vs population size')
# Error Graph
fig , ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(times),cost,'r',label='Error')
ax.set_xlabel('Iterations')
ax.set_ylabel('costs')
ax.set_title('Error vs training Epoch')
# display weights
print(w)
# the prediction value 
pval = float(input('populations = '))
print('The prediction value = ' , w[0,0]+pval*w[0,1])