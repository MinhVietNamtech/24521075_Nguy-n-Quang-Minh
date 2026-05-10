import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

class DecisionTreeClassifier_scratch:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1, max_features = "sqrt"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _build_tree(self, X, y, depth):
        num_samples, _ = X.shape
        # Count labels
        num_labels = np.bincount(y)
        # Majority class
        predicted_class = np.argmax(num_labels)

        node={
            "type": "leaf",
            "class": predicted_class
        }

        # Stop condition
        if (
            depth < (self.max_depth if self.max_depth is not None else np.inf) 
            and num_samples >= self.min_samples_split
            and len(np.unique(y)) > 1
        ):
            best_feat, best_thresh = self._best_split(X, y)

            if best_feat is not None:
                # Perform split
                left_idx = X.iloc[:, best_feat] < best_thresh
                right_idx = ~left_idx

                if (
                    np.sum(left_idx) >= self.min_samples_leaf
                    and np.sum(right_idx) >= self.min_samples_leaf
                ):
                    node={
                        "type": "node",
                        "feature": best_feat,
                        "threshold": best_thresh,
                        "left": self._build_tree(X[left_idx], y[left_idx], depth+1),
                        "right": self._build_tree(X[right_idx], y[right_idx], depth+1)
                    }
        return node

    def _gini(self, y):
        m = len(y)
        if m==0:
            return 0
        probs = np.bincount(y) / m
        return 1.0 - np.sum(probs ** 2)
    
    def _best_split(self, X, y):
        m, n = X.shape
        if m<=1:
            return None, None
        smallest_impurity = float("inf")
        best_feat, best_thresh = None, None
        for feat in range(n):
            # Sort by feature
            sorted_idx = np.argsort(X.iloc[:, feat])
            X_sorted = X.iloc[sorted_idx, feat]
            y_sorted = y.iloc[sorted_idx]

            # Candidate splits: midpoints
            for i in range(1, m):
                if X_sorted.iloc[i] == X_sorted.iloc[i-1]:
                    continue
                impurity = ((i) * self._gini(y_sorted[:i+1]) + (m-i) * self._gini(y_sorted[i+1:])) / m
                if impurity < smallest_impurity:
                    best_feat = feat
                    best_thresh = X_sorted.iloc[i]
                    smallest_impurity = impurity
        return best_feat, best_thresh   

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
    
    def predict(self, X):
        m, n = X.shape
        label = [None] * m
        for i in range(m):
            x = X.iloc[i, :]
            root = self.root
            while root["type"]== "node":
                if(x[root["feature"]] < root["threshold"]):
                    root = root["left"]
                else:
                    root = root["right"]
            label[i] = root["class"]
        return np.array(label)

if __name__ == "__main__":
    red_df = pd.read_csv('wine_quality/winequality-red.csv', sep=';')
    white_df = pd.read_csv('wine_quality/winequality-white.csv', sep=';')
    df = pd.concat([red_df, white_df])
    features = df.columns[:-1]
    label = df.columns[-1]
    X = df[features]
    y = df[label]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    model = DecisionTreeClassifier_scratch(max_depth=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"F1_scratch (DT): {f1_score(y_test, y_pred, average='macro')}")