import numpy as np

class LogisticRegression:
	def __init__(self, lr=0.01, _lambda=0.01):
		self.lr = lr
		self._lambda = _lambda
	
	def __prepare_input(self, X):
		intercept = np.ones((X.shape[0], 1))
		return np.concatenate((intercept, X), axis=1)
	
	def __sigmoid(self, z):
		return 1 / (1 + np.exp(-z))
	
	def __loss(self, h):
		cross_entropy = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
		reg = self._lambda * np.sum(theta ** 2) / (2 * self.y.size)
		return cross_entropy + reg
	
	def bind(self, X, y):
		self.X = self.__prepare_input(X)
		self.y = y
		self.theta = np.zeros(self.X.shape[1])
	
	def train(self, num_iter=100000):
		for i in range(num_iter):
			z = np.dot(self.X, self.theta)
			h = self.__sigmoid(z)
			gradient = (np.dot(self.X.T, h - self.y) + self._lambda * self.theta) / self.y.size
			self.theta -= self.lr * gradient
	
	def predict_prob(self, data):
		data = self.__prepare_input(data)
		return self.__sigmoid(np.dot(data, self.theta))
	
	def predict(self, data, threshold=0.5):
		return self.predict_prob(data) >= threshold