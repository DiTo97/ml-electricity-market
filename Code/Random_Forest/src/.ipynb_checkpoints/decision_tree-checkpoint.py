import numpy as np

class Node:
    """Node class for the regression tree model
    
    Attributes
    ----------
    leaf: bool
        True if the node is a leaf

    child_left: Node
        Left child node

    child_right: Node
        Right child node

    split_feature_id: int
        index of the feature that led to the
        best split from this node

    split_val: int
        threshold value that led to the
        best split for this node

    score: int
        Score associated to each node
        (local model used for the prediction)

    """

    def __init__(self):
        self.leaf = False

        self.child_left = None
        self.child_right = None

        self.split_feature_id = None
        self.split_gain = None
        self.split_val = None

        self.score = None
            
    def split(self, X_instances, Y_instances, depth, params, file=None):
        """Greedy algorithm to split the node and create its
        children recursively by finding the best split
        w.r.t its instance set. (RANDOM FOREST Regression VERSION)
        
        Implementation of the greedy strategy as described in the par.
        'Greedy Solution to Empirical Risk Minimization'
        of the report.

        Parameters
        ----------
        X_instances: np.ndarray
            Instance set of the node

        Y_instances: np.ndarray
            Output of X_instances

        depth: int
            Depth of the node

        params: dict
            Set of building options
        
        file: file handler [optional]
            For debugging purposes

        """

        # file.write('Depth: %d\n' % depth)
        
        # Set score
        self.score = np.mean(Y_instances)
        
        # First STOP criteria 
        # To stop growing, just set the flag leaf

        # If the maximum depth has been reached
        if params['max_depth'] is not None:
            if depth > params['max_depth']:
                self.leaf = True
                return
        
        # If there are less elements than the minimum
        # number accepted to split
        if X_instances.shape[0] < params['min_samples_split']:
            self.leaf = True
            return
        
        # Init of control vars
        best_cost = float('inf')
        best_feature_id = None
        best_val = None
   
        best_left_X_instances = None
        best_right_X_instances = None

        best_left_Y_instances = None
        best_right_Y_instances = None
        
        # Select a random subset of features to split on
        num_features = X_instances.shape[1]
        rnd_features = np.random.permutation(num_features)[0:min(params['max_features'], num_features)] \
                if params['max_features'] is not None \
                else np.arange(num_features)
    
        # Outer minimization: for each feature
        for j in rnd_features:
            # Build the threshold set Tj
            # Extract the column whose ID is feature_id
            # and sort the X instances' values in
            # ascending order. Take, then, the IDs
            # of the sorted X instances.
            sorted_X_instances = np.unique(X_instances[:, j])

            # Control var for inner minimization
            best_inner_cost = float('inf')
            
            # Inner minimization: for each threshold
            for t in sorted_X_instances:     

                # Get the elements of the left region
                left_region_X_instances = X_instances[X_instances[:,j] <= t] 
                left_region_Y_instances = Y_instances[X_instances[:,j] <= t] 
                
                # Get the elements of the right region
                right_region_X_instances = X_instances[X_instances[:,j] > t]               
                right_region_Y_instances = Y_instances[X_instances[:,j] > t]

                # Check if the split is valid: at least a leaf should be generated
                if (left_region_X_instances.shape[0] < params['min_samples_leaf'] or \
                        right_region_X_instances.shape[0] < params['min_samples_leaf']):
                    continue
                
                # Compute costs
                left_cost = self._compute_cost(left_region_Y_instances)
                right_cost = self._compute_cost(right_region_Y_instances)
                
                # Compute the objective function
                cost = (left_region_Y_instances.size/Y_instances.size)*left_cost \
                        + (right_region_Y_instances.size/Y_instances.size)*right_cost
                
                if cost < best_inner_cost:
                    
                    # Update control vars
                    best_inner_cost = cost
                    
                    updt_left_Y_instances = left_region_Y_instances
                    updt_right_Y_instances = right_region_Y_instances

                    updt_left_X_instances = left_region_X_instances
                    updt_right_X_instances = right_region_X_instances

                    updt_val = t

    
            if best_inner_cost < best_cost:

                best_cost = best_inner_cost
                
                # Assignment (j*,t*)
                best_feature_id = j
                best_val = updt_val
                
                # Assignment (DL,DR)
                best_left_X_instances = updt_left_X_instances
                best_left_Y_instances = updt_left_Y_instances

                best_right_X_instances = updt_right_X_instances
                best_right_Y_instances = updt_right_Y_instances
                                
        # file.write('Variable: %d\n' % best_feature_id)
        # file.write('Threshold: %f\n' % best_val)

        # Second STOP criteria        

        # Handle the case in which non valid splits are generated
        # (check for left or right is interchangeable)
        if best_left_X_instances is None:
            self.leaf = True
            return
        

        # file.write('Left instances:\n')
        # np.savetxt(file, best_left_X_instances[:,best_feature_id], newline=" ",fmt='%10.5f')
        # file.write('\n')
        # file.write('Right instances:\n')
        # np.savetxt(file, best_right_X_instances[:,best_feature_id], newline=" ",fmt='%10.5f')
        # file.write('\n')
        # file.write('----------------------\n')
        
        
        # Check if the current node is a leaf according 
        # to the min_split_gain option. Otherwise, grow
        # the tree with two new nodes.
        if best_cost < params['min_split_gain']:
            self.leaf = True
            return
        else:
            # Save the information about the split
            # for the current node
            self.split_feature_id = best_feature_id
            self.split_gain = best_cost
            self.split_val = best_val

            # Start building children nodes (recursively)
            self.left_child = Node()
            self.left_child.build(best_left_X_instances,
                        best_left_Y_instances,depth + 1, params)

            self.right_child = Node()
            self.right_child.build(best_right_X_instances,
                        best_right_Y_instances, depth + 1, params)
            
    def _compute_cost(self,Y_instances):
        """
        Compute cost to find the best split (regression trees
        for random forests)
        
        See the definition of cost function in the par.
        'Greedy Solution to Empirical Risk Minimization'
        of the report.

        Parameters
        ----------
        Y_instances: np.ndarray
            Output set of the node

        """
        if(Y_instances.shape[0] == 0):
            return 0 
        
        return np.mean(np.square(Y_instances - np.mean(Y_instances)))


    def predict(self, X):
        """
        Get the prediction for a single input (used to navigate the tree)

        Parameters
        ----------
        X: np.ndarray
            Single input

        """
        
        # If the node is a leaf, just return the score,
        # otherwise recursively explore the tree until
        # the correct leaf is reached.
        
        if self.leaf:
            return self.score
        else:
            if X[self.split_feature_id] <= self.split_val:
                return self.left_child.predict(X)
            else:
                return self.right_child.predict(X)

            
class Tree():   
    """
    Decision tree class for regression. Used as a sort of private class
    to build random forests.
    
    Attributes
    ----------
    root: Node
    The root node of the tree
    
    """

    def __init__(self):
        self.root = None

    
    def build(self, X_instances, Y_instances, params):
        """
        Start building a decision tree (regression tree
        for random forests)

        Parameters
        ----------
        X_instances: np.ndarray
            Input of the training set
        
        Y_instances: np.ndarray
            Output of the training set
            
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

            max_features: int, default=None
            The number of features to consider when looking for
            the best split

        """
        
        self.root = Node()
        current_depth = 0

        # file = open('RF_testfile.txt','w')
        # file.write('--Start--\n')
        # file.write('\n')
        # file.write('Root ------\n')
        
        self.root.split(X_instances, Y_instances,
                            current_depth, params)

    def predict(self, X):
        """
        Start navigating the tree to get the prediction
        for a single input.

        Parameters
        ----------
        X: np.ndarray
            Single input
            
        """
        # To navigate the tree just call the recursive 
        # function predict of the root node
        
        return self.root.predict(X)
