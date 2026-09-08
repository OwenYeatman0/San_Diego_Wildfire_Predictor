
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, roc_auc_score
from shapely.geometry import Point

def generate_candidate_data(perimeter_geometry, buffer_distance_m = 2000, spacing_m = 250 ):
    buffered = perimeter_geometry.buffer(buffer_distance_m)
    minx, miny, maxx, maxy = buffered.bounds
    
    xs = np.arange(minx, maxx, spacing_m)
    ys = np.arange(miny, maxy, spacing_m)\
    
    points = []
    for x in xs:
        for y in ys:
            pt = Point(x, y)
            if buffered.contains(pt):
                points.append(pt)
    return points



def build_pixel_training_data(sd_daily):
    records = []
    
    for fire_id, group, in sd_daily.groupby('left'):
        group = group.sort_values('event day').reset_index(drop=True)
        for i in range(len(group)-1):
            break
        
    
    
    
    
def last_day_models(training_df, group_col ='fire_id', day_col = 'event_day'):
    training_df = training_df.sort_values([group_col, day_col]).copy()
    training_df['prev_day_growth'] = training_df.groupby(group_col)['dy_ar_km2'].shift(1)
    training_df['prev_day_wind_speed'] = training_df.groupby(group_col)['wind_speed'].shift(1)
    training_df['total_area_currently'] = training_df.groupby(group_col)['dy_ar_km2'].cumsum - training_df['dy-ar_km2']
    
    # first day of the fire
    training_df[['prev_day_growth', 'prev_day_wind_speed']] = training_df[['prev_day_growth', 'prev_day_wind_speed']].fillna(0)
    return training_df


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
        

