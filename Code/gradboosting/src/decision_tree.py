import numpy as np

# Custom constants
from .utils import NUMBER_Infinity

class Node:
    """Node class for the GBDT
    
    Attributes
    ----------
    leaf: bool
        True if the node is a leaf

    child_left: Node
        Left child node

    child_right: Node
        Right child node

    split_feature_id: int
        ID of the feature that led to the
        best split from this node

    split_val: int
        Instance of the node associated to the
        feature whose ID is split_feature_id

    weight: int
        Weight associated to each leaf node when
        its depth exceeds the tree's max depth

    """

    def __init__(self):
        self.leaf = False

        self.child_left = None
        self.child_right = None

        self.split_feature_id = None
        self.split_gain = None
        self.split_val = None

        self.weight = None

    def _calculate_leaf_weight(self, gradient, hessian, lambd):
        """Calculate the optimal weight of this
        leaf node w.r.t. the gradient and 
        the Hessian matrices.
        (Refer to (5) of Reference[1])

        Parameters
        ----------
        gradient: np.ndarray
            Gradient matrix of the node

        hessian: np.ndarray
            Hessian matrix of the node

        lambd: float
            Regularization parameter

        """

        return -np.sum(gradient) / (
                    np.sum(hessian) + lambd)

    def _calculate_loss_reduction(self, G, H, G_l, H_l, G_r, H_r, lambd):
        """Calculate the loss reduction after the split w.r.t. the
        components of the gradient and of the Hessian matrices
        over each child and the node itself, which can be used
        as a scoring function. This score is like the Gini impurity 
        score for evaluating decision trees, except that it is
        derived for a wider range of objective functions.
        (Refer to (7) of Reference[1])

        Please note: Each G (H) parameter actually represents the 
        sum of the gradient (Hessian) components, here. Thus, it is
        a number and not a matrix.

        Parameters
        ----------
        G: float
            Sum of the components of the 
            current node's gradient

        H: float
            Sum of the components of the 
            current node's Hessian

        G_r: float
            Sum of the components of the 
            right child's gradient

        H_r: float
            Sum of the components of the 
            right child's Hessian

        G_l: float
            Sum of the components of the 
            left child's gradient

        H_l: float
            Sum of the components of the 
            left child's Hessian

        lambd: float
            Regularization parameter

        """

        def calculate_score(G, H):
            return np.square(G) / (H + lambd)

        return calculate_score(G_l, H_l) + calculate_score(G_r, H_r) \
                        - calculate_score(G, H)

    def build(self, X_instances, gradient, hessian,
                        shrinkage_rate, depth, params):
        """Greedy algorithm to build the node and its
        children recursively by finding the best split
        w.r.t its instance set.

        Parameters
        ----------
        gradient: np.ndarray
            Gradient matrix of the node

        hessian: np.ndarray
            Hessian matrix of the node

        X_instances: np.ndarray
            Instance set of the node

        shrinkage_rate: float
            Normalization term applied to the weigth
            of each terminal leaf node

        depth: int
            Depth of the node

        params: dict
            Set of building options

        """

        # Assert they have equal shape
        if not (len(X_instances) == len(gradient) \
                    == len(hessian)):
            raise ValueError("The instance set X, the gradient and the Hessian's shapes must match in order to find the best split in the node and build new ones.")

        # If the node's depth is bigger than max_depth
        # or if its X_instances aren't enough to make a split,
        # just tag it as a leaf and calculate its weight
        # in the CART w.r.t gradient, Hessian, and lambda.
        if (depth > params['max_depth']) or (len(X_instances) 
                    < params['min_samples_split']):
            self.leaf = True
            self.weight = self._calculate_leaf_weight(gradient, hessian,
                    params['lambda']) * shrinkage_rate

            return

        # Precompute the sum of the gradient
        # and Hessian matrices' components for later use
        G = np.sum(gradient)
        H = np.sum(hessian) 

        # Init of control vars
        best_gain = -NUMBER_Infinity
        best_feature_id = None
        best_val = 0.

        # Set of IDs of the X instances which will
        # be later passed to the left (right) child
        best_left_X_instances_ids  = None
        best_right_X_instances_ids = None

        num_features = X_instances.shape[1]

        # Select a random subset of IDs of
        # the features to split on, which helps
        # preventing the CART learner from overfitting
        features_ids = np.random.permutation(num_features)[0:min(num_features,
                                int(params['max_features'] * num_features))]

        for feature_id in features_ids:
            # Starting from the left child
            G_l, H_l  = 0., 0.

            # Extract the column whose ID is feature_id
            # and sort the X instances' values in
            # ascending order. Take, then, the IDs
            # of the sorted X instances.
            sorted_X_instances_ids = X_instances[:, feature_id].argsort()

            for j in range(sorted_X_instances_ids.shape[0]):
                X_instance_id = sorted_X_instances_ids[j]

                G_l += gradient[X_instance_id]
                H_l += hessian[X_instance_id]

                # Derive statistics for the 
                # right child from the left child's
                G_r = G - G_l
                H_r = H - H_l

                current_gain = self._calculate_loss_reduction(G, H, G_l, H_l,
                        G_r, H_r, params['lambda'])

                if current_gain > best_gain:
                    current_left_X_instances_ids  = sorted_X_instances_ids[:j+1]
                    current_right_X_instances_ids = sorted_X_instances_ids[j+1:]

                    # Reject if min_samples_leaf is not guaranteed
                    if (len(current_left_X_instances_ids) < params['min_samples_leaf']) \
                                    or (len(current_right_X_instances_ids) < params['min_samples_leaf']):
                        continue

                    # Update control vars
                    best_gain = current_gain
                    best_feature_id = feature_id
                    best_val = X_instances[X_instance_id, feature_id]

                    # Update the IDs of the X instances that will be assigned
                    # to the children nodes that the current one may spawn,
                    # according to the new best gain.
                    best_left_X_instances_ids  = current_left_X_instances_ids
                    best_right_X_instances_ids = current_right_X_instances_ids

        # Check if the current node is a leaf according 
        # to the min_split_gain option. Otherwise, grow
        # the tree with two new nodes.
        
        # Please note: Level-wise tree growth
        if best_gain < params['min_split_gain']:
            self.leaf = True
            self.weight = self._calculate_leaf_weight(gradient, hessian,
                params['lambda']) * shrinkage_rate
        else:
            # Information on splitting
            self.split_feature_id = best_feature_id
            self.split_gain = best_gain
            self.split_val = best_val

            # Build children nodes
            self.left_child = Node()
            self.left_child.build(X_instances[best_left_X_instances_ids],
                        gradient[best_left_X_instances_ids], hessian[best_left_X_instances_ids],
                        shrinkage_rate, depth + 1, params)

            self.right_child = Node()
            self.right_child.build(X_instances[best_right_X_instances_ids],
                        gradient[best_right_X_instances_ids], hessian[best_right_X_instances_ids],
                        shrinkage_rate, depth + 1, params)

    def predict(self, X_i):
        if self.leaf:
            return self.weight
        else:
            if X_i[self.split_feature_id] <= self.split_val:
                return self.left_child.predict(X_i)
            else:
                return self.right_child.predict(X_i)

class Tree():
    """Regression tree for Ensemble learning"""
    def __init__(self):
        self.root = None

    def build(self, X_instances, gradient, hessian,
                    shrinkage_rate, params):
        # Assert they have equal shape
        if not (len(X_instances) == len(gradient) \
                    == len(hessian)):
            raise ValueError("The instance set X, the gradient and the Hessian's shapes must match in order to find the best split in the node and build new ones.")
        
        self.root = Node()
        self.root.build(X_instances, gradient, hessian, shrinkage_rate,
                                depth=0, params=params)

    def predict(self, X_i):
        return self.root.predict(X_i)
