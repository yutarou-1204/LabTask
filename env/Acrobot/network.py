import numpy as np

inodes = 6
hnodes = 3
onodes = 3

N = int(inodes*hnodes + hnodes*onodes)

def save(wbest,G):
	w1 = np.array(wbest[:int(inodes*hnodes)]).reshape(hnodes,inodes).tolist()
	w2 = np.array(wbest[int(inodes*hnodes):]).reshape(onodes,hnodes).tolist()
	with open(f'../../Weights/acrobot_nn_weight_{G}.py',mode='w') as file:
		file.write(f"w1 = {w1}\nw2 = {w2}")
