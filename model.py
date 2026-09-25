import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, roc_auc_score


def compute_positive_weight(y):
    positive = y.sum()
    negative = len(y) - positive
    return negative/positive

def train_and_evaluate(training_df, feature_cols, target_col = 'burned', group_col = 'fire_id', n_splits = 5):
    X = training_df[feature_cols]
    y = training_df[target_col]
    groups = training_df[group_col]
    
    
    pw = compute_positive_weight(y)
    print(f'scale positive weight: {pw:.2f}')
    
    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate (gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model = xgb.XGBClassifier(
                n_estimators = 300, max_depth= 6, learning_rate =  0.05, 
                scale_pos_weight = pw, eval_metric = 'auc', random_state = 1
            )
       
        model.fit(X_train, y_train)
        
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = model.predict(X_test)
        auc = roc_auc_score(y_test, probabilities)
        fold_scores.append(auc)
        print(f'\n Fold {fold+1} - Auc: {auc:.3f} ---')
        print(classification_report(y_test, predictions))
        
        print(f'\n Mean AUC across folds: {np.mean(fold_scores):.3f} '
              f'(range: {min(fold_scores):.3f} - {max(fold_scores):.3f})')
    return fold_scores


def train_final (training_df, feature_cols, target_col = 'burned'):
    X = training_df[feature_cols]
    y = training_df[target_col]
    pw = compute_positive_weight(y)
    
    model = xgb.XGBClassifier(
        n_estimators = 300, max_depth = 6, learning_rate = 0.05,
        scale_pos_weight = pw, eval_metric = 'auc', random_state = 1
        )
    
    model.fit(X, y)
    return model

def get_feature_col_importance(model, feature_cols):
    return pd.DataFrame({
        'feature': feature_cols,
        'importance' : model.feature_importances_}).sort_values('importance', ascending = False)
        

