"""
Training script for Revenue & Churn Optimization Engine
Trains a Random Forest model for churn prediction
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath=None):
    """
    Load customer data from CSV file
    
    Args:
        filepath: Path to the CSV file (if None, uses default relative to project root)
    
    Returns:
        DataFrame with customer data
    """
    if filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        filepath = os.path.join(project_root, 'data', 'customer_data.csv')
    df = pd.read_csv(filepath)
    print(f"Loaded data from {filepath}")
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df):
    """
    Preprocess data for model training
    
    Args:
        df: Raw DataFrame
    
    Returns:
        X: Features DataFrame
        y: Target Series
        label_encoders: Dictionary of label encoders for categorical features
    """
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Drop customer_id as it's not a feature
    if 'customer_id' in df_processed.columns:
        df_processed = df_processed.drop('customer_id', axis=1)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['customer_segment', 'payment_method']
    
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            label_encoders[col] = le
    
    # Separate features and target
    X = df_processed.drop('churned', axis=1)
    y = df_processed['churned']
    
    print(f"\nFeature columns: {list(X.columns)}")
    print(f"Number of features: {len(X.columns)}")
    print(f"\nTarget distribution:")
    print(y.value_counts())
    
    return X, y, label_encoders

def train_model(X_train, y_train, X_test, y_test):
    """
    Train Random Forest model and evaluate performance
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
    
    Returns:
        model: Trained Random Forest model
        metrics: Dictionary of evaluation metrics
    """
    print("\nTraining Random Forest model...")
    
    # Initialize Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return model, metrics

def print_evaluation_metrics(metrics, y_test):
    """
    Print evaluation metrics
    
    Args:
        metrics: Dictionary of evaluation metrics
        y_test: True target values
    """
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print("="*50)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, metrics['y_pred'], 
                               target_names=['Not Churned', 'Churned']))

def save_model(model, filepath=None, label_encoders=None):
    """
    Save trained model and label encoders to pickle files
    
    Args:
        model: Trained model
        filepath: Path to save the model (if None, uses default relative to project root)
        label_encoders: Dictionary of label encoders
    """
    if filepath is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        filepath = os.path.join(project_root, 'models', 'churn_model.pkl')
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {filepath}")
    
    # Save label encoders if provided
    if label_encoders:
        encoders_path = filepath.replace('.pkl', '_encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        print(f"Label encoders saved to {encoders_path}")

def main():
    """
    Main training pipeline
    """
    print("="*50)
    print("REVENUE & CHURN OPTIMIZATION ENGINE - TRAINING")
    print("="*50)
    
    # Load data
    df = load_data()
    
    # Preprocess data
    X, y, label_encoders = preprocess_data(df)
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Train model
    model, metrics = train_model(X_train, y_train, X_test, y_test)
    
    # Print evaluation metrics
    print_evaluation_metrics(metrics, y_test)
    
    # Save model
    save_model(model, filepath=None, label_encoders=label_encoders)
    
    # Feature importance
    print("\n" + "="*50)
    print("TOP 10 FEATURE IMPORTANCE")
    print("="*50)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()
