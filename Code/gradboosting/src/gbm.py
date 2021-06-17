import sys
import time

import numpy as np

from datetime import datetime

# # Parallelism stuff
# from pathos.multiprocessing import ProcessingPool as Pool

# Custom imports
from .decision_tree import Tree

from .utils import calculate_MSE
from .utils import Dataset

# Custom constants
from .utils import NUMBER_Infinity

class GBDT():
    """Gradient boosting class for regression built upon
    CART trees that descend following an L2-gradient.
    
    Attributes
    ----------
    params: dict
        Set of building options

    best_iteration: int
        Iteraton in which the CART learner whose
        validation loss score is lowest was built

    models: array
        Array of CART learners

    """

    def __init__(self):
        # Init default params
        self.params = {
            'gamma': 0.,
            'lambda': 1.,
            'min_split_gain': 0.1,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_depth': 5,
            'max_features': 1.,
            'learning_rate': 0.3
        }

        self.best_iteration = None
        self.models = None

    def _calculate_Y_scores(self, data_set, models):
        """Predict each i-th instance of X given the chosen
        weak models and return the Y score of them all"""

        if len(models) == 0:
            return None
        
        return self.predict(data_set.X, models=models)

    def _calculate_L2_gradient(self, training_set, Y_scores):
        Y = training_set.Y

        # Why all of them to 2? Because deriving
        # the loss function's gradient -2 * (Y - Y_scores)
        # yields a nxD matrix of 2s as an
        # approximation of the Hessian matrix.
        hessian = np.full(len(Y), 2)

        # Generate the first weight as the mean of Y,
        # and its corrisponding gradient, if Y_scores is None,
        # else let it descend following the L2 norm.
        if Y_scores is None:
            Y_scores = np.mean(Y)

        gradient = -2 * (Y - Y_scores)

        return gradient, hessian

    def _calculate_gradient(self, training_set, Y_scores, metric="L2"):
        # L2 norm is supoorted only
        if metric == "L2":
            return self._calculate_L2_gradient(training_set, Y_scores)

    def _calculate_L2_loss(self, models, data_set):
        Y_scores = self._calculate_Y_scores(data_set, models)
        return calculate_MSE(data_set.Y, Y_scores)

    def _calculate_loss(self, models, data_set, metric="L2"):
        """Compute the loss function over the current weak learners
        given the specified metric. The output will be rounded
        to the 6-th decimal to guarantee termination."""
        # L2 loss is supported only
        if metric == "L2":
            loss = self._calculate_L2_loss(models, data_set)

        return round(loss, 6)

    def _build_CART_learner(self, training_set, gradient,
                hessian, shrinkage_rate):
        CART_learner = Tree()
        CART_learner.build(training_set.X, gradient, hessian,
                shrinkage_rate, self.params)

        return CART_learner

    def train(self, params, training_set, num_boost_round=20,
                validation_set=None, early_stopping_rounds=5):
        # Update custom params
        self.params.update(params)

        if self.params['min_samples_split'] < 2 * self.params['min_samples_leaf']:
            raise ValueError('Parameter min_samples_split must be >= 2 * min_samples_leaf, otherwise no split could ever generate two valid leaf nodes.')

        models = []
        shrinkage_rate = 1.

        best_iteration = None
        best_val_loss  = NUMBER_Infinity

        # Microsoft's LightGBM implementation always takes
        # a validation_set into consideration, even when it
        # is equal to None, to compute the best_iteration
        # parameter aside from an actual CV.

        # Thus, I took the same idea.
        if validation_set is None:
            X = training_set.X
            Y = training_set.Y

            # Train/validation XY 70-30 split
            from sklearn.model_selection import train_test_split
            X, X_val, Y, Y_val = train_test_split(X,
                        Y, test_size=0.3)

            training_set = Dataset(X, Y)
            validation_set = Dataset(X_val, Y_val)

        training_start_time = datetime.now()

        print("Training until validation scores don't improve for {} rounds."
                    .format(early_stopping_rounds))

        # Create a num_boost_round number
        # of weak CART learners
        for iter_cnt in range(num_boost_round):
            iter_start_time = datetime.now()

            Y_scores = self._calculate_Y_scores(training_set, models)
            gradient, hessian = self._calculate_gradient(training_set, Y_scores)

            CART_learner = self._build_CART_learner(training_set,
                    gradient, hessian, shrinkage_rate)

            # Update the iter_cnt at each step by
            # the chosen learning_rate to build a different
            # CART_learner until iter_cnt > 0
            if iter_cnt > 0:
                shrinkage_rate *= self.params['learning_rate']

            models.append(CART_learner)

            # Performance errors
            training_loss = self._calculate_loss(models, training_set)
            val_loss = self._calculate_loss(models, validation_set)
            
            if iter_cnt % int(round((early_stopping_rounds / 2))) == 0:
                print("Iteration {:>3}, Training L2 error: {:.10f}, Validation L2 error: {:.10f}, Elapsed time: {}"
                        .format(iter_cnt, training_loss, val_loss, datetime.now() - iter_start_time))

            # Wait for at least the 2nd iteration to
            # assert best_iteration is never 0,
            # otherwise there may be cases where
            # the CART learners will use 0 models!
            if iter_cnt != 0:
                # Update info on validation performance
                # if there's any available
                if val_loss < best_val_loss:
                    best_iteration = iter_cnt
                    best_val_loss  = val_loss

            # Break from the loop if best_iteration hasn't been
            # updated for at least early_stopping_rounds rounds
            if iter_cnt - (best_iteration or 0) >= early_stopping_rounds:
                print("Early stopping. Best iteration is:")
                print("Iteration {:>3}, Training's L2 error: {:.10f}"
                    .format(best_iteration, best_val_loss)); break

        self.models = models
        self.best_iteration = best_iteration

        print("Training finished. Elapsed time: {}\n"
            .format(datetime.now() - training_start_time))

    def _predict_single(self, X_i, models=None, num_iterations=None):
        # # Pool() gave a lot of problems on Windows
        # # because of duplicate children spawns so I chose
        # # not to do it in the end.
        # with Pool() as P:
        #     return sum(P.map(lambda m: m.predict(X_i), models[:num_iterations]))
    
        return np.sum(m.predict(X_i) for m in models[:num_iterations])
    
    def predict(self, X, models=None, num_iterations=None):
        if models is None:
            if self.models is None:
                raise ValueError("Models can't be an empty set")
            models = self.models
            
        Y_pred = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            Y_pred[i] = self._predict_single(X[i], models,
                                num_iterations)
            
        return Y_pred