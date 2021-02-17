import matplotlib.pyplot as plt

def plot(data):
	fig = plt.figure(figsize=(8, 6))
	ax = fig.add_subplot(111)
	ax.plot(data, ".", ms=3, color = "k")
	ax.set_xlabel("Number of evaluations")
	ax.set_ylabel("vale")
	plt.yscale('symlog')
	plt.show()