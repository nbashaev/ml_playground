import numpy as np
from binary_regression import LogisticRegression

class MultipleClassRegression:
	def __init__(self, lr=0.01, num_iter=100000):
		self.lr = lr
		self.num_iter = num_iter
	
	def __convert_labels(self, y):
		self.label_map, encoded_labels = np.unique(y, return_inverse=True)
		self.num_of_classes = len(self.label_map)
		return encoded_labels
	
	def fit(self, X, y):
		y = self.__convert_labels(y)
		self.classifiers = []
		
		for i in range(self.num_of_classes):
			self.classifiers.append( LogisticRegression(lr=self.lr, num_iter=self.num_iter) )
			self.classifiers[i].fit(X, (y == i).astype(int))
	
	def __predict_prob(self, X):
		return np.vstack([classifier.predict_prob(X) for classifier in self.classifiers])
	
	def predict(self, X):
		probs = self.__predict_prob(X)
		return np.apply_along_axis(lambda column: self.label_map[np.argmax(column)], 0, probs)