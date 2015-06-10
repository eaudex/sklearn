#!/usr/bin/python
from sklearn import datasets

def load(filename='./dataset/heart_scale'):
	(X,y) = datasets.load_svmlight_file(f=filename, n_features=None, multilabel=False, zero_based='auto', query_id=False)
	print 'X', X.shape, X.dtype, 'y', y.shape, y.dtype
	return (X,y)

if __name__ == '__main__':
	y,X = load()
