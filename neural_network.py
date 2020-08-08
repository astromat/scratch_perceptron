import numpy as np 
import matplotlib.pyplot as plt
from random import *
import random as rd
from sklearn.datasets import make_circles
import time

#Crear data set
n = 500  #nro de datos
p=2  # numero de caracteristicas de los datos
#generar los datos. X es el input (en este caso una coordenada x,y), Y el output, en este caso una 
#clase 0, o 1 (pertenece o no a un grupo)
X,Y = make_circles(n_samples=n, factor=0.5, noise=0.05)

#print(X)
#plt.scatter(X[Y==0,0],X[Y==0,1], c="skyblue")
#plt.scatter(X[Y==1,0],X[Y==1,1], c="red")
#plt.axis("equal")
#plt.show()

Y = Y[:, np.newaxis]
#print(X.shape)
#print(Y.shape)

class neural_layer():
	"""
	clase solo entrega la estrutura de datos,
	los parametros de la capa: 
	n_conn: nro conexiones que entran
	n_neur: nro de neuronas
	act_f: activation function 
	"""
	def __init__(self, n_conn, n_neur,act_f):
		self.act_f = act_f
		self.bias = np.random.rand(1, n_neur)*2-1 #el 1 indica que es un vect columna
		self.w = np.random.rand(n_conn, n_neur)*2-1



def sigmoid(x):
	func = 1./(1.+np.e**(-x))
	return func

def sigmoide_deriv(x):
	func = x * (1-x)
	return func

sigm = (lambda x: 1./(1.+np.e**(-x)), lambda x: x * (1-x))

tanh = (lambda x: 2./(1.+np.e**(-2*x)) - 1, lambda x: 1 - (2./(1.+np.e**(-2*x)) - 1)**2.0 )



def relufunc(x):
	func = np.maximum(0,x)
	return func

def reluDerivative(x):
	x[x<=0] = 0
	x[x>0] = 1
	return x


def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

relu = (lambda x: ReLU(x) , lambda x: dReLU(x) )

#diseñar la red
"""
layer0 = neural_layer(p, 4, sigmoid)
layer1 = neural_layer(4, 8, sigmoid)
layer2 = ...

e.g., topology = [p, 4,8,16,4,1]
p: numero de entradas iniciales (nro neuronas la primera capa)
4: nro neuronas de la siguiente capa
...
1: ultima neurona
4:
"""

topology = [p, 4,8,16,4,1]

def create_red(topology, act_f):
	"""
	todas las capas tienen la misma funcion de activacion
	"""
	nn = []
	for i, layer in enumerate(topology[:-1]): #loop descartando el último valor
		nn.append(neural_layer(topology[i], topology[i+1], act_f))
	return nn

neural_net = create_red(topology, sigm)

#crear funcion de coste

l2_cost = (lambda Yp, Yr: np.mean((Yp-Yr)**2.), lambda Yp, Yr: (Yp-Yr))

def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):


	"""
	recibe: 
	la arquitectura de la red (neural_net)
	los datos de entrada X, 
	datos salida Y
	funcion costo
	learning rate
	"""
	#forwards pass
	#toma el vector de entrada, y los pasa capa por capa
	out = [(None, X)]

	for i,layer in enumerate(neural_net):
		z = out[-1][1] @ neural_net[i].w +neural_net[i].bias
		a = neural_net[i].act_f[0](z) #a son las capas
		out.append((z,a))
	#recordar: Y son los valores reales, 
	#out[-1][1] son los valores predichos 
	# por la red
	#print(l2_cost(out[-1][1], Y)) #Y 

	if train:
		#backward propagation
		deltas = []
		for i in reversed(range(0, len(neural_net))):

			z = out[i+1][0]
			a = out[i+1][1]
			#print(a.shape)
			if i==len(neural_net)-1:
				#calcular delta ultima capa
				
				deltas.insert(0, l2_cost[1](a,Y)*neural_net[i].act_f[1](a))
				#print(a.shape)
			else:
				deltas.insert(0, deltas[0] @ _w.T * neural_net[i].act_f[1](a))
			_w = neural_net[i].w

		#gradient descent
			neural_net[i].b = neural_net[i].bias - np.mean(deltas[0], axis=0, keepdims=True)*lr
			neural_net[i].w = neural_net[i].w - out[i][1].T @  deltas[0]*lr

	## devolviendo el valor predicho (que esta en la ultima capa)
	return out[-1][1]


#train(neural_net,X,Y,l2_cost,0.5)


# test red 
# if p = 2 => X = np.array([[1, 2], [5, 6], [2, 6]])
# X = np.array([[1], [6], [6]])
# Y = np.array([[1], [6], [6]])
#

#create data
npoint = 50
#theta = np.linspace(-(4 * np.pi), 2 * np.pi, npoint)
theta = np.linspace(0, 2 * np.pi, npoint)


X = []
Y = []

for i in range(len(theta)):
	valueY = [np.cos(theta[i])]
	valueX = [theta[i]]
	Y.append(valueY)
	X.append(valueX)
	i += 1
Y = np.array(Y)
X = np.array(X)
print(X.shape)
print(Y.shape)

p = 1
topology = [p,2,4,2,1]
#neural_n = create_red(topology, sigm)
neural_n = create_red(topology, tanh)

loss = []
for i in range(10000):
	#entrenar red
	pY = train(neural_n, X, Y, l2_cost, lr = 0.001)
	if i %25 ==0:
		loss.append(l2_cost[0](pY,Y))
		_x0 = theta 
		_Y = np.zeros(len(theta))

		for i, x0 in enumerate(_x0):
			#print(i,x0)
			#para hacer una prediccion
			_Y[i] = train(neural_n, np.array([x0]), Y, l2_cost, train=False)[0][0]
			#print(_Y[i])
	

		#plt.plot(range(len(loss)), loss)
		plt.plot(X,Y, 'b', label="real data")
		plt.plot(_x0,_Y, 'r', label="predicted data")
		#plt.legend()
		plt.pause(0.05)
		plt.show(block=False)
		#time.sleep(0.1)
		plt.close("all")
		plt.clf()





#plotting data
plt.plot(X,Y, 'r', label="real data")
plt.plot(X,pY, 'r', label="predicted data")

plt.legend()
plt.show()

quit()

#Probar red neuronal
import time
from IPython.display import clear_output
import gc

p = 2
topology = [p, 4,8,1]

neural_n = create_red(topology, sigm)

loss = []  ## guardar los costes, para poder ver como evoluciona

for i in range(2500):
	#entrenar red
	pY = train(neural_n, X, Y, l2_cost, lr = 0.05)
	if i %25 ==0:
		loss.append(l2_cost[0](pY,Y))
		res = 50 #resolucion para hacer una maya
		_x0 = np.linspace(-1.5, 1.5, res)
		_x1 = np.linspace(-1.5, 1.5, res)
		_Y = np.zeros((res,res))

		for i0, x0 in enumerate(_x0):
			for i1,x1 in enumerate(_x1):
				#para hacer una prediccion
				_Y[i0,i1] = train(neural_n, np.array([x0,x1]), Y, l2_cost, train=False)[0][0]
		

		fig, (ax1, ax2) = plt.subplots(2)

		ax1.pcolormesh(_x0, _x1, _Y, cmap='coolwarm')
		ax1.axis('equal')		

		ax1.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], c= 'skyblue')
		ax1.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], c= 'salmon')
		plt.pause(0.1)
		
		#clear_output(wait=True)
		
		#plt.show(block=False)
		#time.sleep(1)
		#plt.close('all')
		
		ax2.plot(range(len(loss)), loss)
		plt.pause(0.1)
		plt.show(block=False)
		time.sleep(0.5)
		plt.close("all")
		plt.clf()


 



