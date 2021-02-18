import gym
import random
import sys
import copy
import numpy as np
import network
sys.path.append('../../')
from MLP.NN import NN
from Algorithm import ES,GA,PSO,DE

env = gym.make('Acrobot-v1')
inodes = network.inodes
hnodes = network.hnodes
onodes = network.onodes
N = network.N
save = network.save

def game(w):
	observation = env.reset()
	w1 = np.array(w[:int(inodes*hnodes)]).reshape(hnodes,inodes)
	w2 = np.array(w[int(inodes*hnodes):]).reshape(onodes,hnodes)
	nn = NN(inodes,hnodes,onodes,w1,w2)
	total_reward = 0
	while True:
		#env.render()
		n = nn.feedforward(observation)
		action = n.index(max(n))
		observation, reward, done, info = env.step(action)
		total_reward += reward
		if done: break
	return w,total_reward

if __name__ == '__main__':
	alg = GA.GA(game,N,save=save)
	save(alg.train(),'')