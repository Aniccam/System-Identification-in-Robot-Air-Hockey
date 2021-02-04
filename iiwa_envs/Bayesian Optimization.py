from math import sin
from math import pi
from numpy import arange
from numpy import vstack
from numpy import argmax
from numpy import asarray
from numpy.random import normal
from numpy.random import random
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot
from get_sim_data import *
from joblib import Parallel, delayed


# objective function
def objective(x):
	model = Model(x)
	Err, _ = model.get_Err()
	# return (x**2 * sin(5 * pi * x)**6.0) + noise
	return Err

# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)

# probability of improvement acquisition function
def acquisition(X, Xsamples, model):
	# calculate the best surrogate score found so far
	yhat, _ = surrogate(model, X)
	best = min(yhat)
	# calculate mean and stdev via surrogate function
	mu, std = surrogate(model, Xsamples)
	mu = mu[:, 0]
	# calculate the probability of improvement
	probs = norm.cdf((best - mu) / (std+1E-9))
	return probs

# optimize the acquisition function

def opt_acquisition(X, y, model):
	# random search, generate random samples
	Xsamples = get_hyp(10000)
	# Xsamples = Xsamples.reshape(len(Xsamples), 1)
	# calculate the acquisition function for each sample
	scores = acquisition(X, Xsamples, model)
	# locate the index of the largest scores
	ix = np.argmax(scores)
	return Xsamples[ix, :]

# plot real observations vs surrogate function
def plot(X, y, model):
	# scatter plot of inputs and real objective function
	pyplot.scatter(X, y)
	# line plot of surrogate function across domain
	Xsamples = asarray(arange(0, 1, 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples, _ = surrogate(model, Xsamples)
	pyplot.plot(Xsamples, ysamples)
	# show the plot
	pyplot.show()

# sample the domain sparsely with noise

# init hyperparameter
# res0, res12, res45, res67, latf
# 0.2, 1.1, 1, 0.8, 0.01

def get_hyp(num):
	"""

	:param num:
	:return lateral friction, restitution, ori vel , :
	"""
	hyperparams = np.zeros((num, 3))
	# for i in range(num):
	# 	hyperparams[i, 1:7] = [np.random.uniform(0., 1.5) for _ in range(6)]
	# 	hyperparams[i, 0] = np.random.uniform(0, 1)
	# 	hyperparams[i, 7] = np.random.uniform(0, 1)
	for i in range(num):
		hyperparams[i, 0] = np.random.uniform(0.3,1)
		hyperparams[i, 1] = np.random.uniform(0.3,1)
		hyperparams[i, 2] = np.random.uniform(-30,30)
	return hyperparams

ls = []
testls = np.ones((32,64))



X = get_hyp(50)
y = np.zeros((X.shape[0],1))
for i in range(X.shape[0]):
	y[i] = objective(X[i, :])
	print('obj', i, y[i])
# y = Parallel()([objective(x) for x in X])
# reshape into rows and cols
# X = X.reshape(len(X), 1)
# y = y.reshape(len(y), 1)
# define the model
model = GaussianProcessRegressor()
# fit the model
model.fit(X, y)

# perform the optimization process
Lossmean = []
Lossstd = []
Lossmean.append(np.mean(y))
Lossstd.append(np.std(y))
iters = 3500
for i in range(int(iters)):
	# select the next point to sample
	x = opt_acquisition(X, y, model)
	# sample the point
	actual = objective(x)
	print('num', i,'chosen actual', actual)
	# summarize the finding
	est, _ = surrogate(model, [x])
	# print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
	# add the data to the dataset
	X = vstack((X, [x]))
	y = vstack((y, actual))
	Lossmean.append(np.mean(y))
	Lossstd.append(np.std(y))
	# update the model
	model.fit(X, y)

plt.plot(np.linspace(0,iters,iters+1), Lossmean, )
plt.fill_between(np.linspace(0,iters,iters+1), np.array(Lossmean)-2* np.array(Lossstd), np.array(Lossmean)+2* np.array(Lossstd), alpha=0.3)
# plot all samples and the final surrogate function
# plot(X, y, model)
# best result
ix = np.argmin(y)

f = open('testls_iter3500_fit50_acq10000.txt', 'w')
#
f.write('*' * 50)
f.write('parameters')
f.write('*' * 50)
f.write('\n')
for x in X:
	f.write(str(x))
	f.write('\n')
f.write('*' * 50)
f.write('Loss')
f.write('*' * 50)
f.write('\n')
for loss in y:
	f.write(str(loss))
	f.write('\n')
f.write('*' * 50)
f.write('Loss_mean')
f.write('*' * 50)
f.write('\n')
for loss_mean in Lossmean:
	f.write(str(loss_mean))
	f.write('\n')
f.write('*' * 50)
f.write('Loss_std')
f.write('*' * 50)
f.write('\n')
for loss_std in Lossstd:
	f.write(str(loss_std))
	f.write('\n')
f.close()

print('Best Result of X,y=: \n',  *np.array(X[ix]),sep=',')
print(y[ix])
plt.show()
