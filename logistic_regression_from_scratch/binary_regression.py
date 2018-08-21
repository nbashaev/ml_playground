import numpy as np

class LogisticRegression:
	def __init__(self, lr=0.01, num_iter=100000):
		self.lr = lr
		self.num_iter = num_iter
	
	def __prepare_input(self, X):
		intercept = np.ones((X.shape[0], 1))
		return np.concatenate((intercept, X), axis=1)
	
	def __sigmoid(self, z):
		return 1 / (1 + np.exp(-z))
	
	def __loss(self, h, y):
		return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
	
	def fit(self, X, y):
		X = self.__prepare_input(X)
		self.theta = np.zeros(X.shape[1])
		
		for i in range(self.num_iter):
			z = np.dot(X, self.theta)
			h = self.__sigmoid(z)
			gradient = np.dot(X.T, (h - y)) / y.size
			self.theta -= self.lr * gradient
	
	def predict_prob(self, X):
		X = self.__prepare_input(X)
		return self.__sigmoid(np.dot(X, self.theta))
	
	def predict(self, X, threshold=0.5):
		return self.predict_prob(X) >= threshold