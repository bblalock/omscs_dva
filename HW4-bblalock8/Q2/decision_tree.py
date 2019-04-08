from util import entropy, information_gain, partition_classes
import numpy as np 
import ast

class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary or list, as preferred
        #self.tree = []
        self.tree = {}

    def learn(self, X, y):
        # TODO: Train the decision tree (self.tree) using the the sample X and labels y
        # You will have to make use of the functions in utils.py to train the tree
        
        # One possible way of implementing the tree:
        #    Each node in self.tree could be in the form of a dictionary:
        #       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
        #    For example, a non-leaf node with two children can have a 'left' key and  a 
        #    'right' key. You can add more keys which might help in classification
        #    (eg. split attribute and split value)
        def train_tree(X, y):
            X = np.array(X)
            y = np.array(y)
            pred = max(set(y), key=list(y).count)

            # If all the datapoints in X have the same attribute value (X1,....,Xm)
            # If all datapoints in X have same class value y
            if (len([list(i) for i in set(tuple(i) for i in X)]) == 1) or (len(np.unique(y)) == 1):
                # Return a left node that predicts the majority of the class values in Y as output
                # Return a leaf node that predicts y as output
                return {'leaf':True,
                        'pred': pred
                        }
            else:
                # Try all the possible attributes of Xj and threshold t and choose the one j for which IG(Y| Xj,t) is maximized
                max_i_gain = 0
                max_gain_attribute = 0
                max_gain_threshold = 0
                for split_attribute in range(X.shape[1]):
                    for threshold in np.unique(X[:, split_attribute]):
                        X_l, X_r, y_l, y_r = partition_classes(X, y, split_attribute, threshold)
                        i_gain = information_gain(y, [y_l, y_r])
                        if i_gain > max_i_gain:
                            max_i_gain = i_gain
                            max_gain_attribute = split_attribute
                            max_gain_threshold = threshold
                            X_left, X_right, y_left, y_right = (X_l, X_r, y_l, y_r)

                node = {'information_gain': max_i_gain,
                        'split_attribute': max_gain_attribute,
                        'threshold': max_gain_threshold
                        }

                if len(y_left) == 0:
                    node['left'] = {'leaf': True,'pred': pred}
                else:
                    node['left'] = train_tree(X_left, y_left)

                if len(y_right) == 0:
                    node['right'] = {'leaf': True,'pred': pred}
                else:
                    node['right'] = train_tree(X_right, y_right)

                return node

        self.tree = train_tree(X, y)

    def classify(self, record):
        # TODO: classify the record using self.tree and return the predicted label
        def predict(tree, record):
            if 'leaf' in tree:
                return tree['pred']

            # print(tree)

            if str(record[tree['split_attribute']]) <= str(tree['threshold']):
                if 'left' in tree:
                    return predict(tree['left'], record)
            else:
                return predict(tree['right'], record)

        prediction = predict(self.tree, record)

        return prediction
