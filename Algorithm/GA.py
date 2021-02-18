import numpy as np
import random
import copy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import Parameters.GA as GA
from Plot import plot

fbest = -1*sys.maxsize
wbest = []

u_lim = GA.u_lim
l_lim = GA.l_lim

alpha = GA.alpha
P = GA.parents
C = GA.children
L = GA.elites
G = GA.G

class GA:
	def __init__(self,func,N,**kwargs):
		self.game = func
		self.R = 1/N
		self.w_list = [[random.uniform(l_lim/10,u_lim/10) for i in range(N)] for j in range(C)]
		self.save = kwargs['save']

	class Parent:
		def __init__(self, w, f):
			self.w = w
			self.f = f

	def _mutation(self,r_list):
		for n in range(len(r_list)):
			if self.R > random.random(): r_list[n] = random.uniform(l_lim,u_lim)
		return r_list

	def elite_judge(self,p_list,l_list):
		r_list = copy.copy(l_list)
		for p in p_list:
			r_list = sorted(r_list,key=lambda r:r.f,reverse=True)
			if r_list[L-1].f < p.f: r_list[L-1] = p
		return r_list

	def blx_alpha(self,j_list):
		r_list = random.sample(j_list,2)
		A = copy.copy(r_list[0].w)
		B = copy.copy(r_list[1].w)
		C = []
		for a,b in zip(A,B):
			L = abs(a-b)
			La = L*alpha
			lower = l_lim if b-La < l_lim else b-La
			upper = u_lim if a+La > u_lim else a+La
			C.append(random.uniform(lower,upper))
		return self._mutation(C)
			
	def train(self):
		global fbest,wbest
		p_list,f_list = [],[]
		l_list = [self.Parent([],-1*sys.maxsize) for i in range(L)]

		for i in range(G):
			for w in self.w_list:
				p = self.game(w)
				if p[1] > fbest:
					fbest = p[1]
					wbest = p[0]
				p_list.append(self.Parent(p[0],p[1]))
				f_list.append(p[1])

			# 評価値の高い順にソート
			p_list = sorted(p_list,key=lambda p: p.f,reverse=True)
			
			# エリート更新
			l_list = self.elite_judge(p_list,l_list)
			print(f"{i+1}: ",max(f_list))

			# 淘汰
			del p_list[P:]

			# エリートと親のリスト
			join_list = p_list + l_list

			# 新たな子のリストを作成
			self.w_list.clear()
			for _ in range(C):
				c = self.blx_alpha(join_list)
				self.w_list.append(c)
			
			p_list.clear()

			if i == int(G/10) -1: self.save(wbest,i+1)
			if i == int(G/2) -1: self.save(wbest,i+1)
			
		print(fbest)
		#plot(f_list)
		return wbest


