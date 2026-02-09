"""
Flood Prediction Model Training Script
Uses Kaggle flood prediction dataset to train an XGBoost classifier
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
import joblib
import json

def download_dataset():
    """Download dataset from Kaggle"""
    try:
        import kagglehub
        path = kagglehub.dataset_download("naiyakhalid/flood-prediction-dataset")
        print(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Using synthetic data for training...")
        return None

def create_synthetic_data():
    """Create synthetic flood data if Kaggle download fails"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'rainfall': np.random.uniform(0, 300, n_samples),  # mm
        'temperature': np.random.uniform(15, 40, n_samples),  # Celsius
        'humidity': np.random.uniform(30, 100, n_samples),  # %
        'wind_speed': np.random.uniform(0, 50, n_samples),  # km/h
        'river_level': np.random.uniform(1, 10, n_samples),  # meters
        'soil_moisture': np.random.uniform(0, 100, n_samples),  # %
    }
    
    df = pd.DataFrame(data)
    
    # Create flood labels based on conditions
    df['flood'] = (
        (df['rainfall'] > 150) & 
        (df['humidity'] > 70) & 
        (df['river_level'] > 6)
    ).astype(int)
    
    # Add some randomness
    flip_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    df.loc[flip_indices, 'flood'] = 1 - df.loc[flip_indices, 'flood']
    
    return df

def train_model():
    """Train the flood prediction model"""
    print("=" * 50)
    print("FLOOD PREDICTION MODEL TRAINING (XGBoost)")
    print("=" * 50)
    
    # Try to download dataset
    dataset_path = download_dataset()
    
    if dataset_path:
        # Look for CSV files in the dataset
        csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]
        if csv_files:
            df = pd.read_csv(os.path.join(dataset_path, csv_files[0]))
            print(f"Loaded dataset: {csv_files[0]}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
        else:
            print("No CSV files found, using synthetic data")
            df = create_synthetic_data()
    else:
        df = create_synthetic_data()
    
    # Prepare features - adjust based on actual dataset columns
    feature_cols = ['rainfall', 'temperature', 'humidity', 'wind_speed', 'river_level', 'soil_moisture']
    target_col = 'flood'
    
    # Check if columns exist, otherwise use available numeric columns
    available_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col not in df.columns:
        # Try to find a suitable target column
        possible_targets = ['FloodProbability', 'Flood', 'flood_risk', 'target']
        for pt in possible_targets:
            if pt in df.columns:
                target_col = pt
                break
        else:
            # Use last column as target
            target_col = available_cols[-1]
    
    feature_cols = [c for c in available_cols if c != target_col]
    
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    print(f"Target column: {target_col}")
    
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    # Convert to binary if needed
    if y.nunique() > 2:
        threshold = y.median()
        y = (y > threshold).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model with hyperparameter tuning
    print("\nTraining XGBoost model with RandomizedSearchCV...")
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 15),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 0.5)
    }
    
    search = RandomizedSearchCV(
        xgb, 
        param_distributions=param_dist, 
        n_iter=20, 
        scoring='accuracy', 
        cv=3, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    
    search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {search.best_params_}")
    model = search.best_estimator_
    
    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Save model and scaler
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save stats to JSON
    stats = {
        'training_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'feature_importance': {col: float(val) for col, val in zip(feature_cols, model.feature_importances_)}
    }
    with open(os.path.join(model_dir, 'model_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"Stats saved to model_stats.json")

    # Overwrite the existing model so the app picks it up
    joblib.dump(model, os.path.join(model_dir, 'flood_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_cols.pkl'))
    
    print(f"\n✅ Model saved to {model_dir}")
    print("=" * 50)
    
    return model, scaler, feature_cols

if __name__ == "__main__":
    train_model()
