from one_vs_all import MultipleClassRegression
from sklearn import datasets, model_selection

def calc_acc(model, X, y):
	return (model.predict(X) == y).astype(int).mean()

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

one_vs_all_regr = MultipleClassRegression(lr=0.1, num_iter=300000)
one_vs_all_regr.fit(X_train, y_train)

print('one_vs_all train acc: {}'.format(calc_acc(one_vs_all_regr, X_train, y_train)))
print('one_vs_all test acc: {}'.format(calc_acc(one_vs_all_regr, X_test, y_test)))