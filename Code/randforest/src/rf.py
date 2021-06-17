import sys
import time

import numpy as np

from datetime import datetime

# Custom imports
from .decision_tree import Tree

from .utils import calculate_MSE


class RandomForestRegressor():
    """
    Random Forest class for REGRESSION
    
    Attributes
    ----------
    params: dict
        Set of building options

        max_depth: int, default=None
        The maximum depth of the tree. If None, then nodes are expanded 
        until all leaves are pure or until all leaves contain less than
        min_samples_split samples.

        min_samples_split: int, default=2
        The minimum number of samples required to split an internal node.

        min_samples_leaf: int, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it produces
        at least min_samples_leaf training samples in each of the left
        and right branches.

        min_impurity_decrease: float, default=0.0
        A node will be split if this split induces a decrease of the 
        impurity greater than or equal to this value.

        n_estimators: int, default=10
        The number of trees in the forest.

        max_features: int, default=None
        The number of features to consider when looking for
        the best split


    models: array
        Array of CART learners (decision trees which make up the forest)

    """

    def __init__(self):
        
        # Init default params
        self.params = {
            'min_split_gain': 0.0,
            'max_depth': None,
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 10,
            'max_features': None
        }

        self.models = None


    def _build_CART_learner(self, X_instances, Y_instances):
        """
        Create a CART learner (regression tree)
        

        Parameters
        ----------
        X_instances: np.ndarray
        Training set input

        Y_instances: np.ndarray
        Training set Output

        """

        # Create a Tree object (from decision_tree.py)
        CART_learner = Tree()
        
        # Start the building of the tree
        CART_learner.build(X_instances, Y_instances, self.params)

        return CART_learner


    def train(self, X_train, Y_train, X_val=None, Y_val=None, params=None):
        """
        Train a random forest regression model.
        Note: If the validation set is not specified, a random 70-30 splitting
        is automatically done on the training set.
        

        Parameters
        ----------
        X_train: np.ndarray
        Training set input

        Y_train: np.ndarray
        Training set Output

        X_val: np.ndarray [optional]
        Validation set input
        
        Y_val: np.ndarray [optional]
        Validation set output

        params: dict
            Set of building options

            max_depth: int, default=None
            The maximum depth of the tree. If None, then nodes are expanded 
            until all leaves are pure or until all leaves contain less than
            min_samples_split samples.
            
            n_estimators: int, default=10
            The number of trees in the forest.
            
            max_features: int, default=None
            The number of features to consider when looking for
            the best split

            min_samples_split: int, default=2
            The minimum number of samples required to split an internal node.

            min_samples_leaf: int, default=1
            The minimum number of samples required to be at a leaf node.
            A split point at any depth will only be considered if it produces
            at least min_samples_leaf training samples in each of the left
            and right branches.

            min_impurity_decrease: float, default=0.0
            A node will be split if this split induces a decrease of the
            objective function greater than or equal to this value.


        """

        # Set the specified parameters if any
        if params is not None:
            self.params.update(params)
            
        print("Model parameters:")
        print(self.params)

        # Initialize the list that will contain the trees
        models = []

        # If the validation set is not specified, use a random
        # 70 - 30 % splitting (non repeatable random choice)
        np.random.seed(None)
        if (X_val is None) or (Y_val is None):
            from sklearn.model_selection import train_test_split
            X_train, X_val, Y_train, Y_val = train_test_split(X_train,
                        Y_train, test_size=0.3)
            
        num_rows = X_train.shape[0]

        training_start_time = datetime.now()

        print("Training started...")

        # Create the forest
        for iter_cnt in range(self.params['n_estimators']):
            
            # Bagging
            # Get a random permutation of the index with replacement
            rnd_idx = np.random.choice(num_rows,
                        size=num_rows, replace=True)
            # Get a random sample of the dataset with replacement
            X_train_bagging = X_train[rnd_idx]
            Y_train_bagging = Y_train[rnd_idx]

            iter_start_time = datetime.now()
                       
            # Create a single tree
            CART_learner = self._build_CART_learner(X_train_bagging, Y_train_bagging)

            # Append the tree to the list
            models.append(CART_learner)

            # Measure performance
            training_loss = self._calculate_loss(X_train_bagging, Y_train_bagging, CART_learner)
            val_loss = self._calculate_loss(X_val, Y_val, CART_learner)

            print("Tree number {:>3}, Training L2 error: {:.10f}, Validation L2 error: {:.10f}, Elapsed time: {}"
                        .format(iter_cnt+1, training_loss, val_loss, datetime.now() - iter_start_time))


        # Update
        self.models = models

        print("Training finished. Elapsed time: {}\n"
            .format(datetime.now() - training_start_time))
        



    def _predict_single_tree(self, X, tree):
        """
        Predict each i-th output of a dataset X
        according to a single decision tree
            
            
        Attributes
        ----------
        X: np.ndarray
        Input dataset

        tree: Tree
        Regression tree       
            
        """

        # Initialize
        Y_scores = np.zeros(len(X))

        # Obtain the prediction for each sample
        # of the dataset
        for i in range(len(X)):
            
            # Cast to positive integers! (because of OUR output)
            Y_scores[i] = max(0,round(tree.predict(X[i])))


        return Y_scores
    



    def _calculate_loss(self, X, Y, tree):
        """
        Compute the loss function (Mean Squared Error)
        on a dataset with respect to the predictions 
        obtained from a single regression tree.
            
            
        Attributes
        ----------
        X: np.ndarray
        Input dataset

        Y: np.ndarray
        True output of the dataset X

        tree: Tree
        Regression tree to be used for predicting the
        output of X       
            
        """

        # Get the prediction
        Y_scores = self._predict_single_tree(X, tree)
        
        # function imported from utils.py
        return calculate_MSE(Y, Y_scores)




    def predict(self, X):
        """
        Predict each i-th output of a dataset X
        according to the random forest model.
            
        See par. 'From CART models to Random Forests'
        of the report

        Attributes
        ----------
        X: np.ndarray
        Input dataset      
            
        """

        # Initialize
        Y_scores = np.zeros(len(X))

        # For each tree of the forest
        for tree in self.models:

            # Accumulate the prediction
            Y_scores += self._predict_single_tree(X, tree)

        # Return the mean
        return Y_scores / len(self.models)

        
       
