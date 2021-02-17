import numpy as np

def relu(x):
	return np.maximum(x,0)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def softsign(x):
	return np.array([n/1+abs(n) for n in x.tolist()])

class NN:
	def __init__(self,inodes,hnodes,onodes,w1,w2):
		self.inodes = inodes
		self.hnodes = hnodes
		self.onodes = onodes

		self.w1 = w1
		self.w2 = w2

	def feedforward(self,idata):
		h0 = idata

		h0_u1 = np.dot(self.w1,h0)
		h1 = sigmoid(h0_u1)

		h1_o = np.dot(self.w2,h1)
		o = sigmoid(h1_o)

		return o.tolist()

	def feedforward_(self,idata):
		h0 = idata

		h0_u1 = np.dot(self.w1,h0)
		h1 = softsign(h0_u1)

		h1_o = np.dot(self.w2,h1)
		o = softsign(h1_o)

		return o.tolist()