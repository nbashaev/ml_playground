import numpy as np

class LogisticRegression:
	def __init__(self, lr, _lambda):
		self.lr = lr
		self._lambda = _lambda
	
	def __sigmoid(self, z):
		return 1 / (1 + np.exp(-z))
	
	def __loss(self, h):
		cross_entropy = (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
		reg = (self._lambda / (2 * self.y.size)) * np.sum(theta ** 2)
		return cross_entropy + reg
	
	def bind(self, X, y):
		self.X = X
		self.y = y
		self.theta = np.zeros(self.X.shape[1])
	
	def train(self, num_iter):
		for i in range(num_iter):
			h = self.__sigmoid(np.dot(self.X, self.theta))
			gradient = (np.dot(self.X.T, h - self.y) + self._lambda * self.theta) / self.y.size
			self.theta -= self.lr * gradient
	
	def predict_prob(self, data):
		return self.__sigmoid(np.dot(data, self.theta))
	
	def predict(self, data):
		return self.predict_prob(data) >= 0.5