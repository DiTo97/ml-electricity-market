import sys
import time

import numpy as np

from datetime import datetime

# Custom imports
from .decision_tree import Tree

from .utils import calculate_MSE
from .utils import Dataset

# Constant vars
NUMBER_Infinity = sys.maxsize 

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
            'max_depth': 5,
            'learning_rate': 0.3
        }

        self.best_iteration = None
        self.models = None

    def _calculate_Y_scores(self, data_set, models):
        """Predict each i-th instance of X given the chosen
        weak models and return the Y score of them all"""

        if len(models) == 0:
            return None

        X = data_set.X
        Y_scores = np.zeros(len(X))

        for i in range(len(X)):
            Y_scores[i] = self.predict(X[i], models=models)

        return Y_scores

    def _calculate_L2_gradient(self, training_set, Y_scores):
        Y = training_set.Y

        # Why all of them to 2? Because deriving the
        # objective function 2 * (Y - Y_scores)
        # yields a nxD matrix of 2s as a really rough
        # approximation of the Hessian matrix.
        hessian = np.full(len(Y), 2)

        # Generate a random gradient if Y_scores is None,
        # else let it descend following the L2 norm.
        if Y_scores is None:
            gradient = np.random.uniform(size=len(Y))
        else:
            gradient = 2 * (Y - Y_scores)


        return gradient, hessian

    def _calculate_gradient(self, training_set, Y_scores, metric="L2"):
        # L2 norm is supoorted only
        if metric == "L2":
            return self._calculate_L2_gradient(training_set, Y_scores)

    def _calculate_L2_loss(self, models, data_set):
        Y_scores = self._calculate_Y_scores(data_set, models)
        return calculate_MSE(data_set.Y, Y_scores)

    def _calculate_loss(self, models, data_set, metric="L2"):
        # L2 loss is supported only
        if metric == "L2":
            return self._calculate_L2_loss(models, data_set)

    def _build_CART_learner(self, training_set, gradient,
                hessian, shrinkage_rate):
        CART_learner = Tree()
        CART_learner.build(training_set.X, gradient, hessian,
                shrinkage_rate, self.params)

        return CART_learner

    def train(self, params, training_set, num_boost_round=20,
                validation_set=None, early_stopping_rounds=5):
        self.params.update(params)

        models = []
        shrinkage_rate = 1.

        best_iteration = None
        best_val_loss  = NUMBER_Infinity

        # TODO: Should it be optional?
        if validation_set is None:
            X = training_set.X
            Y = training_set.Y

            # Train/validation XY split
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

            print("Iteration {:>3}, Training's L2 error: {:.10f}, Validation's L2 error: {:.10f}, Elapsed time: {}"
                        .format(iter_cnt, training_loss, val_loss, datetime.now() - iter_start_time))

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

    def predict(self, X, models=None, num_iterations=None):
        if models is None:
            if self.models is None:
                raise ValueError("Models can't be an empty set")
            models = self.models
        
        return np.sum(m.predict(X) for m in models[:num_iterations])
