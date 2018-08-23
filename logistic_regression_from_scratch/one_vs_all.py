import numpy as np
from binary_regression import LogisticRegression

class MultipleClassRegression:
	def __init__(self, lr=0.01, _lambda=0.01):
		self.lr = lr
		self._lambda = _lambda
	
	def __convert_labels(self, y):
		self.label_map, encoded_labels = np.unique(y, return_inverse=True)
		self.num_of_classes = len(self.label_map)
		return encoded_labels
	
	def bind(self, X, y):
		y = self.__convert_labels(y)
		self.classifiers = []
		
		for i in range(self.num_of_classes):
			self.classifiers.append( LogisticRegression(lr=self.lr, _lambda=self._lambda) )
			self.classifiers[i].bind(X, (y == i).astype(int))
	
	def train(self, num_iter):
		for i in range(self.num_of_classes):
			self.classifiers[i].train(num_iter)
	
	def __predict_prob(self, data):
		return np.vstack([classifier.predict_prob(data) for classifier in self.classifiers])
	
	def predict(self, data):
		probs = self.__predict_prob(data)
		return np.apply_along_axis(lambda column: self.label_map[np.argmax(column)], 0, probs)