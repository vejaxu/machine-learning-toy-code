import numpy as np

class DecisionTree():
    def __init__(self) -> None:
        self.tree = None

    def entropy(self, y):
        classes = np.unique(y)
        entropy = 0
        for cls in classes:
            p = np.mean(y == cls)
            entropy -= p * np.log2(p)
        return entropy
    
    def information_gain(self, X, y, feature_idx, threshold):
        left_indices = X[:, feature_idx] <= threshold
        right_indices = ~left_indices
        n = len(y)
        left_entropy = self.entropy(y[left_indices])
        right_entropy = self.entropy(y[right_indices])
        left_weight = np.sum(left_indices) / n
        right_weight = np.sum(right_indices) / n
        return self.entropy(y) - (left_weight * left_entropy + right_weight * right_entropy)
    
    def find_best_split(self, X, y):
        best_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self.information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        return best_feature_idx, best_threshold
    
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return {"class": y[0]}
        
        best_feature_idx, best_threshold = self.find_best_split(X, y)
        left_indices = X[:, best_feature_idx] <= best_threshold
        right_indices = ~left_indices
        left_subtree = self._grow_tree(X[left_indices], y[left_indices])
        right_subtree = self._grow_tree(X[right_indices], y[right_indices])
        return {'feature_idx': best_feature_idx, 'threshold': best_threshold, 'left': left_subtree, 'right': right_subtree}
    

    def _predict_one(self, x, tree):
        if 'class' in tree:
            return tree['class']
        feature_value = x[tree['feature_idx']]
        if feature_value <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])
        
    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]
    

if __name__ == "__main__":
    # Sample data
    X = np.array([[2, 3], [1, 5], [3, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])

    # Initialize and train decision tree model
    model = DecisionTree()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)
    print("Predictions:", y_pred)