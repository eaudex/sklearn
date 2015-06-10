#!/usr/bin/python
from sklearn import grid_search
from sklearn import svm, linear_model
import load_data_in_libsvm

def search(X, y, learner, grid, scorer):
	# grid search
	searcher = grid_search.GridSearchCV(estimator=learner, param_grid=grid, scoring=scorer, refit=True, fit_params=None, n_jobs=1, cv=5, verbose=0)
	searcher.fit(X,y)
	print 'scorer', searcher.scorer_

	print 'CV Results'
	for grid_point in searcher.grid_scores_:
		print '\t', grid_point
	print 'best cv parameter', searcher.best_params_
	print 'best cv score', searcher.best_score_
	print 'best cv estimator', searcher.best_estimator_

	# call methods on the estimator with the best found parameter
	print 'train score', searcher.score(X,y)
	#searcher.decision_function(X)
	#searcher.predict(X)
	#searcher.predict_proba(X)

	return searcher.best_estimator_


if __name__ == '__main__':
	# load data
	(X,y) = load_data_in_libsvm.load()

	# build learner
	# LIBSVM: SVC
	#learner = svm.SVC(tol=1e-3, max_iter=-1, cache_size=1000, class_weight=None, shrinking=True, probability=False, verbose=0,random_state=0)
	#grid = {'kernel':['rbf'], 'C':[2**i for i in range(-1,1)], 'gamma':[2**i for i in range(-1,1)]}
	# LIBLINEAR: LR
	learner = linear_model.LogisticRegression(tol=1e-4, fit_intercept=True,intercept_scaling=1.0, class_weight=None, random_state=0,dual=False)
	grid = {'penalty':['l1','l2'], 'C':[2**i for i in range(-1,1)]}

	# metrics
	#scorer = 'accuracy'
	#scorer = 'log_loss'
	scorer = 'roc_auc'

	search(X, y, learner, grid, scorer)


