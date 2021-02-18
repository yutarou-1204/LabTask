import numpy as np
import random
import copy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import Parameters.ES as ES
from Plot import plot

fbest = -1*sys.maxsize
wbest = []

u_lim = ES.u_lim
l_lim = ES.l_lim
u_mulim = ES.u_mulim
l_mulim = ES.l_mulim
P = ES.parent
C = ES.children
G = ES.G

class ES:
	def __init__(self,func,N,**kwargs):
		self.game = func
		self.w_list = [[random.uniform(l_lim/10,u_lim/10) for i in range(N)] for j in range(C)]
		self.p_list = []
		self.f_list = []
		self.save = kwargs['save']

	class Parent:
		def __init__(self,w,f):
			self.w = w
			self.f = f

		def mutation(self):
			re_list = copy.copy(self.w)
			return [n+random.uniform(l_mulim,u_mulim) for n in re_list]

	def train(self):
		global fbest,wbest
		for i in range(G):
			for w in self.w_list:
				p = self.game(w)
				if p[1] > fbest:
					fbest = p[1]
					wbest = p[0]
				self.p_list.append(self.Parent(p[0],p[1]))
				self.f_list.append(p[1])

			#評価の良い順にソート
			self.p_list = sorted(self.p_list,key=lambda p: p.f,reverse=True)
			print(f"{i+1}: ",max(self.f_list))

			#淘汰
			del self.p_list[P:]

			#新たな子の生成
			self.w_list = [self.p_list[random.randint(0,P-1)].mutation() for i in range(C)]

			if i == int(G/10) -1: self.save(wbest,i+1)
			if i == int(G/2) -1: self.save(wbest,i+1)

		print(fbest)
		plot(self.f_list)
		return wbest


