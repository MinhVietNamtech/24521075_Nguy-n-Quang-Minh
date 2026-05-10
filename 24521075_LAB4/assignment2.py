import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import assignment1

class RandomForestClassification_scratch:
    def __init__(
        self,
        n_estimators = 100,
        max_depth = 10,
        min_samples_split = 2,
        min_samples_leaf = 1,
        max_features = "sqrt"
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
    
    def _bootstrap(self, X, y):
        m, n = X.shape
        idxs = np.random.choice(m, m, replace=True)
        feat = np.random.choice(n, int(np.sqrt(n)) if self.max_features=='sqrt' else 3, replace=False)
        
        return X.iloc[idxs, feat], y.iloc[idxs]
    
    def fit(self, X, y):
        self.forest = []
        for _ in range(self.n_estimators):
            tree = assignment1.DecisionTreeClassifier_scratch(
                max_depth= self.max_depth,
                min_samples_split = self.min_samples_split,
                min_samples_leaf = self.min_samples_leaf,
                max_features = self.max_features 
            )
            X_sample, y_sample = self._bootstrap(X, y)
            tree.fit(X_sample, y_sample)
            self.forest.append(tree)
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.forest])
        return np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis = 0,
            arr=tree_preds
        )

if __name__ == "__main__":
    red_df = pd.read_csv('wine_quality/winequality-red.csv', sep=';')
    white_df = pd.read_csv('wine_quality/winequality-white.csv', sep=';')
    df = pd.concat([red_df, white_df])
    features = df.columns[:-1]
    label = df.columns[-1]

    X, y = df[features], df[label]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=True,
        random_state=42
    )

    model = RandomForestClassification_scratch(n_estimators=10, max_depth=100, min_samples_leaf=1, min_samples_split=2)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"F1_scratch (RF): {f1_score(y_test, y_pred, average='macro')}")