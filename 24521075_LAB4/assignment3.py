import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

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

    tree = DecisionTreeClassifier(max_depth=100)
    tree.fit(X_train, y_train)
    y_tree = tree.predict(X_test)

    forest = RandomForestClassifier(n_estimators=100, max_depth=10)
    forest.fit(X_train, y_train)
    y_forest = forest.predict(X_test)

    print(f"F1 (DecisionTreeClassifier): {f1_score(y_test, y_tree, average='macro')}")
    print(f"F1 (RandomForestClassifier): {f1_score(y_test, y_forest, average='macro')}")