"""
Quick Enhanced Model Training
Optimized for speed while still improving accuracy
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def download_dataset():
    try:
        import kagglehub
        path = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
        print(f"✅ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def train_quick():
    print("=" * 60)
    print("🌊 QUICK ENHANCED FLOOD MODEL TRAINING")
    print("=" * 60)
    
    # Load dataset
    dataset_path = download_dataset()
    if not dataset_path:
        print("❌ Could not download dataset")
        return
    
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
    df = pd.read_csv(os.path.join(dataset_path, csv_files[0]))
    
    print(f"📊 Loaded {len(df)} samples")
    
    # Setup features and target
    available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    target_col = 'FloodProbability'
    feature_cols = [c for c in available_cols if c != target_col]
    
    print(f"📈 Features: {feature_cols}")
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # Binary classification
    threshold = y.median()
    y_binary = (y > threshold).astype(int)
    print(f"🎯 Binary threshold: {threshold:.2f}")
    print(f"   Class 0: {(y_binary == 0).sum()}, Class 1: {(y_binary == 1).sum()}")
    
    # Add engineered features
    if 'MonsoonIntensity' in feature_cols and 'TopographyDrainage' in feature_cols:
        X['MonsoonDrainage'] = X['MonsoonIntensity'] * (10 - X['TopographyDrainage']) / 10
    if 'Urbanization' in feature_cols and 'DrainageSystems' in feature_cols:
        X['UrbanDrainageRisk'] = X['Urbanization'] * (10 - X['DrainageSystems']) / 10
    if 'Deforestation' in feature_cols and 'ClimateChange' in feature_cols:
        X['EnvironmentalRisk'] = (X['Deforestation'] + X['ClimateChange']) / 2
    
    feature_cols = X.columns.tolist()
    print(f"✨ With engineered features: {len(feature_cols)} total")
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models with optimized params
    models = {
        'Random Forest (Optimized)': RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    
    best_model = None
    best_accuracy = 0
    best_name = None
    
    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        train_acc = model.score(X_train_scaled, y_train)
        test_acc = model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        cv_mean = cv_scores.mean()
        
        # ROC-AUC
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"   Training Accuracy:  {train_acc:.4f}")
        print(f"   Test Accuracy:      {test_acc:.4f}")
        print(f"   CV Accuracy:        {cv_mean:.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"   ROC-AUC:            {roc_auc:.4f}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_model = model
            best_name = name
    
    # Feature importance
    print(f"\n📊 Feature Importance ({best_name}):")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.head(10).iterrows():
        bar = '█' * int(row['importance'] * 40)
        print(f"   {row['feature']:30s} {row['importance']:.4f} {bar}")
    
    # Save model
    model_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(best_model, os.path.join(model_dir, 'flood_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))
    
    # Final evaluation
    y_pred = best_model.predict(X_test_scaled)
    print(f"\n📋 Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred))
    
    print("\n" + "=" * 60)
    print("🎯 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Model:        {best_name}")
    print(f"Test Accuracy:     {best_accuracy:.4f} ({best_accuracy*100:.1f}%)")
    print(f"Model saved to:    {model_dir}")
    print("=" * 60)
    
    return best_model, scaler, feature_cols

if __name__ == "__main__":
    train_quick()
