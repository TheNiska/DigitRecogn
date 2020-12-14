import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import time
start = time.time()

ALFA_ZERO = 0.009
BETA = 0.70
LAMBD = 0
EPSILON = 0.00000000001
K_ITER = 400
cost_func = []
x_iter = []

data = pd.read_csv('train.csv', delimiter=',')
y = data[['label']]
y = y.to_numpy()
y = y.T           # 1 x 42000

form = np.ones((10,1))
y = y * form

isZero = y[0,:] == 0
y[0,:] = (y[0,:] + 1) * isZero

for j in range(1,10):
	isNumber = y[j,:] == j
	y[j,:] = (y[j,:] * isNumber) / j
print(y.shape)
y_whole = y
y = y[:,:34000]   # 1 x 34000




x = data.drop('label', axis=1)
x = x.to_numpy()
x = x.T         # 784 x 42000


x_whole = x
x = x[:,:34000]

n0 = 784
n1 = 12
n2 = 6
n3 = 10

np.random.seed(1)

m = 34000

a0 = x

w1 = np.random.randn(n1,n0) * math.sqrt(1/n0) # xavier initialization
b1 = np.random.randn(n1,1)

w2 = np.random.randn(n2,n1) * math.sqrt(1/n1)
b2 = np.random.randn(n2,1)  

w3 = np.random.randn(n3,n2) * math.sqrt(1/n2)
b3 = np.random.randn(n3,1)

mini_batch_size = 256

for o in range(K_ITER):
	proc = (o / K_ITER) * 100
	print( '{:.2f}'.format(proc), '%')
	ALFA = (1 / (1 + 0.005*o))*ALFA_ZERO  # learning rate decay
	t = 0

	while t < 34000:
		y_mini = y[:,t:t+mini_batch_size]
		a_mini = a0[:,t:t+mini_batch_size]
		m_mini = y_mini.shape[1]

		z1 = np.dot(w1, a_mini) + b1
		a1 = np.tanh(z1)
	    
		z2 = np.dot(w2, a1) + b2
		a2 = np.tanh(z2)
			    
		z3 = np.dot(w3, a2) + b3
		ti = math.e**z3

		J = (1 / m_mini) * (- np.sum(y_mini * np.log(a3)))

		dz3 = a3 - y_mini
		dw3 = (1/m_mini) * (np.dot(dz3, a2.T))
		db3 = np.sum(dz3, axis=1, keepdims=True) * (1/m_mini)
			    
		dz2 = np.dot(w3.T, dz3) * (1 - np.power(a2, 2))
		dw2 = (1/m_mini) * (np.dot(dz2, a1.T))
		db2 = np.sum(dz2, axis=1, keepdims=True) * (1/m_mini)
			    
		dz1 = np.dot(w2.T, dz2) * (1 - np.power(a1, 2))
		dw1 = (1/m_mini) * (np.dot(dz1, a_mini.T))
		db1 = np.sum(dz1, axis=1, keepdims=True) * (1/m_mini)

		w3 = w3 - ALFA * dw3 
		b3 = b3 - ALFA * db3
			    
		w2 = w2 - ALFA * dw2 
		b2 = b2 - ALFA * db2

		w1 = w1 - ALFA * dw1
		b1 = b1 - ALFA * db1   
		
		Jtemp = '{:.4f}'.format(J)

		cost_func.append(float(Jtemp))
		x_iter.append(o)

		t += mini_batch_size
	print(J)

fig = plt.subplots()  
plt.plot(x_iter, cost_func)
plt.show()

#-----VALIDATION---------------------------------------------------------------------------------

y_dev = y_whole[:,34000:]
x_dev = x_whole[:,34000:]

z1 = np.dot(w1, x_dev) + b1
a1 = np.tanh(z1)

z2 = np.dot(w2, a1) + b2 
a2 = np.tanh(z2)
    
z3 = np.dot(w3, a2) + b3
ti = math.e**z3
a3 = ti / np.sum(ti, axis=0)
np.set_printoptions(precision=4, suppress=True)
print(a3[:,7995:])
print(np.argmax(a3[:,7995:], axis=0))

J_dev = (1 / 8000) * (- np.sum(y_dev * np.log(a3)))
print('Cost function in dev examples: ', J_dev)

#-------------------------------------------------------------------------------------------------

end = time.time()
print('RunTime = ' + '{:.2f}'.format((end-start)/60))