import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, auc


def split_salary_label(df):
    """Split data from label"""
    df['salary'] = np.where(df['salary']==" 50000+.", 1, 0)
    y = df['salary']
    X = df.drop(['salary'], axis=1)
    return y, X

def one_hot_encode_factors(df):
    """One-hot encode string columns"""
    one_hot_df = pd.get_dummies(df.select_dtypes(include=['object']))
    numeric_df = df.select_dtypes(exclude=['object'])
    return numeric_df.join(one_hot_df)

def accuracy(pred, actual):
    """Returns percentage of correctly classified labels"""
    return sum(pred==actual) / len(actual)


def main():
    # Import data and split out labels
    train = pd.read_csv('data/train_data_balanced.csv')  # balanced training data
    train_full = pd.read_csv('data/train_data_full.csv') # full training data
    val = pd.read_csv('data/validation_set.csv')  # balanced training data
    val_pos = pd.read_csv('data/validation_set_positive.csv')  # balanced training data
    train_large = pd.concat([train, val_pos])
    test = pd.read_csv('data/test_set.csv')

    # Split into labels and data
    y_train, X_train = split_salary_label(train)
    X_train = one_hot_encode_factors(X_train)

    y_train_full, X_train_full = split_salary_label(train_full)
    X_train_full = one_hot_encode_factors(X_train_full)

    y_val, X_val = split_salary_label(val)
    X_val = one_hot_encode_factors(X_val)

    y_train_large, X_train_large = split_salary_label(train_large)
    X_train_large = one_hot_encode_factors(X_train_large)

    y_test, X_test = split_salary_label(test)
    X_test = one_hot_encode_factors(X_test)

    # Optimize RF parameters using grid search
    rf = RandomForestClassifier(criterion='entropy')
    param_grid = {
        'n_estimators': [100], 
        'max_depth': [1, 5, 20],
        'max_features': [10, 20, 30],
        'n_jobs': [30],
        'min_samples_leaf': [3, 5, 10]
    }

    grid_rf = GridSearchCV(rf, param_grid, cv=2, verbose=3)
    grid_rf.fit(X_train, y_train)
    print("#-------- DONE WITH GRID SEARCH.")
    best_model = grid_rf.best_estimator_
    best_params = grid_rf.best_params_ 
    scores = grid_rf.grid_scores_
    print(best_params)

    # Run with chosen parameters
    rf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_features=30, max_depth=20,
    	                            min_samples_leaf=3, bootstrap=True, oob_score=True, n_jobs=30, random_state=0)
    # rf = rf.fit(X_train_full, y_train_full)
    rf = rf.fit(X_train, y_train)

    # Validation set performance
    y_val_pred = rf.predict(X_val)
    y_val_prob = rf.predict_proba(X_val)
    print("Validation Accuracy = %f" % accuracy(y_val_pred, y_val))
    print("Log loss = %f" % log_loss(y_val, y_val_prob))
    print("AUC ROC score = %f" % roc_auc_score(y_val, [p[1] for p in y_val_prob]))

    # Retrain on full data
    # rf = rf.fit(X_train_large, y_train_large)

    # Test set performance
    y_test_pred = rf.predict(X_test)
    y_test_prob = rf.predict_proba(X_test)
    print("Test Accuracy = %f" % accuracy(y_test_pred, y_test))
    print("Log loss = %f" % log_loss(y_test, y_test_prob))
    print("AUC ROC score = %f" % roc_auc_score(y_test, [p[1] for p in y_test_prob]))

    # Plot ROC curve
    fpr, tpr, threshold = roc_curve(y_test, [p[1] for p in y_test_prob])
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    plt.title('ROC Curve for Test Data')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    fig.savefig('figures/test_roc.png', bbox_inches='tight')

    # Save feature importance data
    importance = rf.feature_importances_
    importance = pd.DataFrame(importance, index=X_train.columns, columns=["Importance"])
    importance["stddev"] = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
    importance = importance[importance['Importance']>0.01]
    importance.to_csv('results/feature_importance.csv')


if __name__ == "__main__":
    main()