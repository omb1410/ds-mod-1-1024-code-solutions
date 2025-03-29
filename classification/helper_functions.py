# TODO: 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def grid_train_random_forest(X, y, params, n_folds=4, eval_metric='accuracy'):
    """
    Train Random Forest binary classifier using a grid of hyperparameters. Return
    the best model according to the specified metric.

    Args:
        X: Array-like of shape (n_samples,n_features) - Training feature data.
        y: Array-like of shape (n_samples,) - Training target data.
        params: Dictionary - Parameter grid on which to perform cross validation.
        n_folds: int - Number of folds to use for cross validation.
        eval_metric: str - Metric to use for evaluating model performance in cross validation.

    Returns:
        model: Best Random Forest model according to evaluation metric.

    Examples:
        model = grid_train_random_forest(X, y, params, 4, "accuracy")
    """

    # TODO: Implement this function
    rfc = RandomForestClassifier()

    gscv_rfc = GridSearchCV(estimator=rfc, param_grid=params, cv=n_folds, scoring=eval_metric)

    gscv_rfc.fit(X, y)

    return gscv_rfc.best_estimator_


def calc_roc_metrics(X, y, model):
    """
    Calculate False Positive Rate (FPR), True Positive Rate (TPR), and Area Under ROC Curve (AUC)
    for a given binary classification model and test data.

    Args:
        X: Array-like of shape (n_samples,n_features) - Test feature data.
        y: Array-like of shape (n_samples,) - Test target data.
        model: Scikit-learn style binary classification model.

    Returns:
        fpr: np.array[float] - False Positive Rates.
        tpr: np.array[float] - True Positive Rates.
        auc: float - Area Under ROC Curve.

    Examples:
        fpr, tpr, auc = calc_roc_metrics(X, y, model)
    """

    # TODO: Implement this function
    y_proba = model.predict_proba(X)[:,1]

    if y.dtype == 'object' or isinstance(y[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)

    fpr, tpr, thresholds = roc_curve(y, y_proba)

    auc = roc_auc_score(y, y_proba)

    return fpr, tpr, auc

def train_xgboost(X_train, y_train, X_test, y_test, params, n_round):
    """
    Train an XGBoost model with the given parameters and train/test data.

    Args:
        X_train: Array-like of shape (n_train_samples,n_features) - Train feature data.
        y_train: Array-like of shape (n_train_samples,) - Train target data.
        X_test: Array-like of shape (n_test_samples,n_features) - Test feature data.
        y_test: Array-like of shape (n_test_samples,) - Test target data.
        params: Dictionary - Parameters to pass into XGBoost trainer.
        n_round: int - Number of rounds of training.

    Returns:
        model: Trained XGBoost model.

    Examples:
        model = calc_roc_metrics(X_train, y_train, X_test, y_test, params)
    """

    # TODO: Implement this function

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    xgbc = xgb.XGBClassifier(**params, n_estimators= n_round)
    
    xgbc.fit(X_train, y_train)

    y_pred_proba = xgbc.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print("ROC AUC Score:", auc)

    return xgbc

    


