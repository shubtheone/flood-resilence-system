"""
Enhanced Flood Prediction Model Training Script
Features:
- Multiple algorithms (Random Forest, Gradient Boosting, XGBoost)
- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust evaluation
- Feature engineering and importance analysis
- Detailed classification metrics
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")

def download_dataset():
    """Download dataset from Kaggle"""
    try:
        import kagglehub
        path = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
        print(f"✅ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 Using synthetic data for training...")
        return None

def load_kaggle_data(dataset_path):
    """Load and preprocess Kaggle flood dataset"""
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    if not csv_files:
        return None
    
    df = pd.read_csv(os.path.join(dataset_path, csv_files[0]))
    print(f"📊 Loaded dataset: {csv_files[0]}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    return df

def create_synthetic_data(n_samples=5000):
    """Create enhanced synthetic flood data if Kaggle download fails"""
    np.random.seed(42)
    
    # Create realistic distributions
    data = {
        'MonsoonIntensity': np.random.uniform(0, 10, n_samples),
        'TopographyDrainage': np.random.uniform(0, 10, n_samples),
        'RiverManagement': np.random.uniform(0, 10, n_samples),
        'Deforestation': np.random.uniform(0, 10, n_samples),
        'Urbanization': np.random.uniform(0, 10, n_samples),
        'ClimateChange': np.random.uniform(0, 10, n_samples),
        'Siltation': np.random.uniform(0, 10, n_samples),
        'AgriculturalPractices': np.random.uniform(0, 10, n_samples),
        'IneffectiveDisasterPreparedness': np.random.uniform(0, 10, n_samples),
        'DrainageSystems': np.random.uniform(0, 10, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create flood probability based on weighted features
    flood_score = (
        df['MonsoonIntensity'] * 0.25 +
        (10 - df['TopographyDrainage']) * 0.15 +
        (10 - df['RiverManagement']) * 0.15 +
        df['Deforestation'] * 0.10 +
        df['Urbanization'] * 0.10 +
        df['ClimateChange'] * 0.10 +
        df['Siltation'] * 0.05 +
        df['IneffectiveDisasterPreparedness'] * 0.05 +
        (10 - df['DrainageSystems']) * 0.05
    )
    
    # Normalize to 0-100
    df['FloodProbability'] = (flood_score / flood_score.max()) * 100
    
    return df

def engineer_features(df, feature_cols):
    """Create additional engineered features"""
    X = df[feature_cols].copy()
    
    # Add interaction features if we have enough samples
    if len(X) > 1000:
        # Risk composite scores
        if 'MonsoonIntensity' in feature_cols and 'TopographyDrainage' in feature_cols:
            X['MonsoonDrainage'] = X['MonsoonIntensity'] * (10 - X['TopographyDrainage']) / 10
        
        if 'Urbanization' in feature_cols and 'DrainageSystems' in feature_cols:
            X['UrbanDrainageRisk'] = X['Urbanization'] * (10 - X['DrainageSystems']) / 10
        
        if 'Deforestation' in feature_cols and 'ClimateChange' in feature_cols:
            X['EnvironmentalRisk'] = (X['Deforestation'] + X['ClimateChange']) / 2
    
    return X

def train_with_grid_search(X_train, y_train, X_test, y_test, algorithm='random_forest'):
    """Train model with hyperparameter tuning"""
    
    if algorithm == 'random_forest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
    elif algorithm == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'min_samples_split': [2, 5],
            'subsample': [0.8, 1.0]
        }
        
    elif algorithm == 'xgboost' and HAS_XGBOOST:
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    print(f"\n🔍 Grid Search for {algorithm}...")
    print(f"   Parameter grid: {len(param_grid)} parameters")
    
    # Use stratified k-fold for more robust evaluation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        model, param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    return best_model, grid_search.best_params_, grid_search.best_score_

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    print(f"\n📊 Evaluation Results for {model_name}")
    print("=" * 50)
    
    # Training and test accuracy
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"CV Accuracy:       {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # ROC-AUC if binary classification
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC Score:     {roc_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'cv_accuracy': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

def train_model_enhanced(use_grid_search=True, algorithms=['random_forest']):
    """Enhanced model training with multiple options"""
    print("=" * 60)
    print("🌊 ENHANCED FLOOD PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Download or create dataset
    dataset_path = download_dataset()
    
    if dataset_path:
        df = load_kaggle_data(dataset_path)
        if df is None:
            df = create_synthetic_data()
    else:
        df = create_synthetic_data()
    
    print(f"\n📈 Dataset Info:")
    print(f"   Total samples: {len(df)}")
    
    # Identify target and features
    available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Find target column
    target_candidates = ['FloodProbability', 'Flood', 'flood', 'target']
    target_col = None
    for tc in target_candidates:
        if tc in df.columns:
            target_col = tc
            break
    
    if target_col is None:
        target_col = available_cols[-1]
    
    feature_cols = [c for c in available_cols if c != target_col]
    
    print(f"   Features: {feature_cols}")
    print(f"   Target: {target_col}")
    
    # Prepare data
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # Convert to binary classification if needed
    if y.nunique() > 2:
        threshold = y.median()
        y = (y > threshold).astype(int)
        print(f"   Converted to binary (threshold: {threshold:.2f})")
    
    print(f"   Class distribution: {dict(y.value_counts())}")
    
    # Feature engineering
    X = engineer_features(df, feature_cols)
    feature_cols = X.columns.tolist()
    print(f"   Engineered features: {feature_cols}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    best_model = None
    best_accuracy = 0
    best_algorithm = None
    best_params = None
    all_results = {}
    
    for algorithm in algorithms:
        print(f"\n{'='*60}")
        print(f"🚀 Training {algorithm.upper()}")
        print("="*60)
        
        try:
            if use_grid_search:
                model, params, cv_score = train_with_grid_search(
                    X_train_scaled, y_train, X_test_scaled, y_test, algorithm
                )
                print(f"\n✅ Best Parameters: {params}")
                print(f"   Best CV Score: {cv_score:.4f}")
            else:
                # Quick training without grid search
                if algorithm == 'random_forest':
                    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
                elif algorithm == 'gradient_boosting':
                    model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
                elif algorithm == 'xgboost' and HAS_XGBOOST:
                    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
                else:
                    continue
                
                model.fit(X_train_scaled, y_train)
                params = model.get_params()
            
            # Evaluate
            results = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, algorithm)
            all_results[algorithm] = results
            
            # Track best model
            if results['test_accuracy'] > best_accuracy:
                best_accuracy = results['test_accuracy']
                best_model = model
                best_algorithm = algorithm
                best_params = params
                
        except Exception as e:
            print(f"❌ Error training {algorithm}: {e}")
            continue
    
    # Feature importance for best model
    if hasattr(best_model, 'feature_importances_'):
        print(f"\n📊 Feature Importance ({best_algorithm}):")
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in importance.iterrows():
            bar = '█' * int(row['importance'] * 50)
            print(f"   {row['feature']:30s} {row['importance']:.4f} {bar}")
    
    # Save the best model
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\n💾 Saving best model ({best_algorithm})...")
    joblib.dump(best_model, os.path.join(model_dir, 'flood_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))
    
    # Save training results
    results_summary = {
        'best_algorithm': best_algorithm,
        'best_params': best_params,
        'training_accuracy': all_results[best_algorithm]['train_accuracy'],
        'test_accuracy': all_results[best_algorithm]['test_accuracy'],
        'cv_accuracy': all_results[best_algorithm]['cv_accuracy'],
        'feature_cols': feature_cols,
        'all_results': all_results
    }
    joblib.dump(results_summary, os.path.join(model_dir, 'training_results.pkl'))
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 TRAINING SUMMARY")
    print("=" * 60)
    print(f"Best Algorithm:     {best_algorithm}")
    print(f"Test Accuracy:      {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")
    print(f"CV Accuracy:        {all_results[best_algorithm]['cv_accuracy']:.4f}")
    print(f"Model saved to:     {model_dir}")
    print("=" * 60)
    
    return best_model, scaler, feature_cols, results_summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Flood Prediction Model')
    parser.add_argument('--quick', action='store_true', help='Quick training without grid search')
    parser.add_argument('--all', action='store_true', help='Train all available algorithms')
    args = parser.parse_args()
    
    algorithms = ['random_forest']
    if args.all:
        algorithms = ['random_forest', 'gradient_boosting']
        if HAS_XGBOOST:
            algorithms.append('xgboost')
    
    train_model_enhanced(
        use_grid_search=not args.quick,
        algorithms=algorithms
    )
