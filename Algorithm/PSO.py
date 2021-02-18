import numpy as np

import copy
import random
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import Parameters.PSO as PSO
from Plot import plot

fbest = -1*sys.maxsize
wbest = []

mi = PSO.min
ma = PSO.max
evalmax = PSO.evalmax
P = PSO.particles
G = PSO.G
w = PSO.w
Cp = PSO.Cp
Cg = PSO.Cg

class PSO:
	def __init__(self,func,N,**kwargs):
		self.game = func
		self.save = kwargs['save']
		self.gbest = []
		self.f = -1*sys.maxsize
		self.particles = [self.Particle([random.uniform(mi/10,ma/10) for i in range(N)]) for j in range(P)]
		
	class Particle:
		def __init__(self,x):
			self.pbest = []
			self.f = -1*sys.maxsize
			self.x = np.array(x)
			self.v = np.zeros(len(x))
		
		def move(self,gbest):
			Rp,Rg = random.uniform(0,1),random.uniform(0,1)
			self.v = w*self.v + Cp*Rp*(self.pbest-self.x) + Cg*Rg*(gbest-self.x)
			self.x = np.clip(self.v + self.x,mi,ma)

	def update(self,x,f):
		self.gbest = x
		self.f = f

	def train(self):
		global fbest,wbest
		f_list = []
		for i in range(G):
			for p in self.particles:
				f = self.game(p.x)
				f_list.append(f[1])
				if f[1] > fbest:
					fbest = f[1]
					wbest = f[0]
				if p.f < f[1]:
					p.f = f[1]
					p.pbest = copy.copy(p.x)
					if self.f < p.f: self.update(p.x,p.f)
			for p in self.particles: p.move(self.gbest)

			print(f"{i+1}: {max(f_list)}")
			
			#ネットワーク保存
			if i == int(G/10) -1: self.save(wbest,i+1)
			if i == int(G/2) -1: self.save(wbest,i+1)

		print(fbest)
		plot(f_list)
		return wbest


