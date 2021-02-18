import numpy as np

def relu(x):
	return np.maximum(x,0)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def softsign(x):
	return np.array([n/1+abs(n) for n in x.tolist()])

def relu(x):
	return np.array([n if n > 0 else 0 for n in x.tolist()])

class NN:
	def __init__(self,w1,w2,**kwargs):
		self.w1 = w1
		self.w2 = w2

		if kwargs['func'] == 'softsign': 
			self.af = softsign
		elif kwargs['func'] == 'relu': 
			self.af = relu
		else: 
			self.af = sigmoid

	def feedforward(self,idata):
		h0 = idata

		h0_u1 = np.dot(self.w1,h0)
		h1 = self.af(h0_u1)

		h1_o = np.dot(self.w2,h1)
		o = self.af(h1_o)

		return o.tolist()
