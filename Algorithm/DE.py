import numpy as np
import random
import copy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import Parameters.DE as DE

from Plot import plot

fbest = -1*sys.maxsize
Wbest = []

mi = DE.min
ma = DE.max
G = DE.G
P = DE.population
F = DE.F
CR = DE.CR

class DE:
	def __init__(self,func,N,**kwargs):
		self.game = func
		self.w_list = [[random.uniform(mi/10,ma/10) for i in range(N)] for j in range(P)]
		self.save = kwargs['save']
		
	class Population:
		def __init__(self,w,f):
			self.w = w
			self.f = f

	def _donor(self,p_list):
		d_list = []
		for i,p in enumerate(p_list):
			choices = [n for n in range(P)]
			choices.remove(i)
			a,b,c = random.sample(choices,3)
			d_list.append(list(np.array(p_list[a].w)+F*(np.array(p_list[b].w)-np.array(p_list[c].w))))
		return d_list

	def _generate(self,p_list):
		N = len(p_list[0].w)
		c_list = [[0 for i in range(N)] for j in range(P)]
		d_list = self._donor(p_list)
		choices = [random.randint(0,N-1) for i in range(P)]
		for i,k in enumerate(choices):
			c_list[i][k] = d_list[i][k]
		c_list = [[c_list[i][k] if k==choices[i] else d_list[i][k] if CR > random.random() else p_list[i].w[k] for k in range(N)] for i in range(P)]
		c_list = [[c_list[i][k] if mi <= c_list[i][k] <= ma else ma if c_list[i][k]>0 else mi for k in range(N)] for i in range(P)]
		return c_list

	def train(self):
		global fbest,wbest
		#初期の子のリスト生成
		p_list,c_list,f_list,g_list = [],[],[],[]
		for i in range(G):
			for w in self.w_list:
				p = self.game(w)
				if p[1] > fbest:
					fbest = p[1]
					wbest = p[0]
				c_list.append(self.Population(p[0],p[1]))
				f_list.append(p[1])
			g_list.append(max(f_list))
			
			print(f"{i+1}: {max(f_list)}")

			if i < 1:
				p_list = copy.copy(c_list)
			else:
				p_list = [p_list[n] if p_list[n].f > c_list[n].f else c_list[n] for n in range(P)]
			
			self.w_list = self._generate(p_list)

			c_list.clear()

			#ネットワーク保存
			if i == int(G/10) -1: self.save(wbest,i+1)
			if i == int(G/2) -1: self.save(wbest,i+1)

		print(fbest)
		plot(f_list)
		return wbest
