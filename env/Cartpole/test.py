import gym
import random
import sys
import copy
import numpy as np
import network
sys.path.append('../../')
from MLP.NN import NN
import Weights.cartpole_nn_weight_ as w

env = gym.make('CartPole-v0')

time = 1000

w1 = w.w1
w2 = w.w2
nn = NN(w1,w2,func='softsign')
for episode in range(10):
	observation = env.reset()
	score = 0
	for i in range(time):
		env.render()
		n = nn.feedforward(observation)
		action = n.index(max(n))
		observation, reward, done, info = env.step(action)
		score += reward
		if reward < 1: break
	print(score)
env.close()