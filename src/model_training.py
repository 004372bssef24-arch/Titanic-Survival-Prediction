"""
Model Training Module
=====================
Trains and evaluates multiple ML models for Titanic survival prediction.
"""

import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from .config import config
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model training and evaluation class.
    
    Trains:
        - Logistic Regression
        - Random Forest
        - Voting Ensemble (combines both)
    """
    
    def __init__(self):
        """Initialize model trainer"""
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.results = {}
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for modeling (encoding, scaling).
        
        Args:
            df: DataFrame with engineered features
        
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Preparing features for modeling...")
        
        # Separate target
        target_col = config.data.get('target_column', 'Survived')
        y = df[target_col]
        
        # Get feature columns from config
        feature_cols = list(config.features.get('selected_features', []))

        # Add engineered features if missing
        for extra_feature in ['FamilySize', 'IsAlone']:
            if extra_feature not in feature_cols:
                feature_cols.append(extra_feature)
        # Filter to only available columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        
        # Encode categorical variables
        # Encode Sex
        if 'Sex' in X.columns:
            self.label_encoders['Sex'] = LabelEncoder()
            X['Sex_Encoded'] = self.label_encoders['Sex'].fit_transform(X['Sex'])
            X.drop('Sex', axis=1, inplace=True)
            feature_cols = [col for col in feature_cols if col != 'Sex']
            feature_cols.append('Sex_Encoded')
            logger.info("  - Encoded Sex -> Sex_Encoded")
        
        # One-hot encode Embarked
        if 'Embarked' in X.columns:
            embarked_dummies = pd.get_dummies(X['Embarked'], prefix='Embarked')
            X = pd.concat([X, embarked_dummies], axis=1)
            X.drop('Embarked', axis=1, inplace=True)
            feature_cols.extend(embarked_dummies.columns.tolist())
            logger.info("  - One-hot encoded Embarked")
        
        # Update feature columns
        self.feature_columns = [col for col in X.columns if col != target_col]
        
        # Reorder columns for consistency
        X = X[self.feature_columns]
        
        # Scale numerical features
        numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        numerical_present = [col for col in numerical_features if col in X.columns]
        
        if numerical_present:
            X[numerical_present] = self.scaler.fit_transform(X[numerical_present])
            logger.info(f"  - Scaled numerical features: {numerical_present}")
        
        logger.info(f"  - Final feature set: {len(self.feature_columns)} features")
        logger.info(f"  - Features: {self.feature_columns}")
        
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
        """
        test_size = config.data.get('test_size', 0.2)
        random_state = config.data.get('random_state', 42)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y  # Maintain class distribution
        )
        
        logger.info(f"Data split complete:")
        logger.info(f"  - Training: {len(self.X_train)} samples ({self.y_train.mean()*100:.1f}% survival)")
        logger.info(f"  - Testing: {len(self.X_test)} samples ({self.y_test.mean()*100:.1f}% survival)")
    
    def train_logistic_regression(self) -> LogisticRegression:
        """
        Train Logistic Regression model with hyperparameter tuning.
        
        Returns:
            Trained Logistic Regression model
        """
        logger.info("\n" + "=" * 50)
        logger.info("Training Logistic Regression...")
        logger.info("=" * 50)
        
        # Get hyperparameters from config
        lr_config = config.models.get('logistic_regression', {})
        param_grid = {
            'C': lr_config.get('C', [0.1, 1, 10]),
            'penalty': lr_config.get('penalty', ['l2']),
            'solver': lr_config.get('solver', ['liblinear', 'lbfgs'])
        }
        
        # Base model
        lr = LogisticRegression(
            max_iter=lr_config.get('max_iter', 1000),
            random_state=lr_config.get('random_state', 42)
        )
        
        # Grid search for best parameters
        cv_folds = config.models.get('random_forest', {}).get('cv_folds', 5)
        grid_search = GridSearchCV(
            lr, param_grid, 
            cv=cv_folds, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate
        y_pred = best_model.predict(self.X_test)
        y_proba = best_model.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
        self.results['Logistic Regression'] = metrics
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
        self.models['logistic_regression'] = best_model
        return best_model
    
    def train_random_forest(self) -> RandomForestClassifier:
        """
        Train Random Forest model with hyperparameter tuning.
        
        Returns:
            Trained Random Forest model
        """
        logger.info("\n" + "=" * 50)
        logger.info("Training Random Forest...")
        logger.info("=" * 50)
        
        # Get hyperparameters from config
        rf_config = config.models.get('random_forest', {})
        param_grid = {
            'n_estimators': rf_config.get('n_estimators', [100, 200]),
            'max_depth': rf_config.get('max_depth', [10, 15, None]),
            'min_samples_split': rf_config.get('min_samples_split', [2, 5]),
            'min_samples_leaf': rf_config.get('min_samples_leaf', [1, 2])
        }
        
        # Base model
        rf = RandomForestClassifier(
            random_state=rf_config.get('random_state', 42),
            n_jobs=-1
        )
        
        # Grid search
        cv_folds = rf_config.get('cv_folds', 5)
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Evaluate
        y_pred = best_model.predict(self.X_test)
        y_proba = best_model.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
        self.results['Random Forest'] = metrics
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"  - {row['feature']}: {row['importance']:.4f}")
        
        self.models['random_forest'] = best_model
        return best_model
    
    def train_voting_ensemble(self) -> VotingClassifier:
        """
        Train Voting Ensemble combining all models.
        
        Returns:
            Trained Voting Ensemble model
        """
        logger.info("\n" + "=" * 50)
        logger.info("Training Voting Ensemble...")
        logger.info("=" * 50)
        
        # Get ensemble config
        ensemble_config = config.models.get('voting_ensemble', {})
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=[
                ('lr', self.models.get('logistic_regression')),
                ('rf', self.models.get('random_forest'))
            ],
            voting=ensemble_config.get('voting', 'soft'),
            weights=ensemble_config.get('weights', [1, 1])
        )
        
        # Train on full training data
        voting_clf.fit(self.X_train, self.y_train)
        
        # Evaluate
        y_pred = voting_clf.predict(self.X_test)
        y_proba = voting_clf.predict_proba(self.X_test)[:, 1]
        
        metrics = self._calculate_metrics(self.y_test, y_pred, y_proba)
        self.results['Voting Ensemble'] = metrics
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        
        self.models['voting_ensemble'] = voting_clf
        return voting_clf
    
    def _calculate_metrics(self, y_true, y_pred, y_proba) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Returns:
            DataFrame with comparison results
        """
        comparison = pd.DataFrame(self.results).T
        comparison = comparison.round(4)
        
        logger.info("\n" + "=" * 50)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 50)
        print(comparison.to_string())
        
        # Identify best model
        best_model = comparison['accuracy'].idxmax()
        logger.info(f"\n🏆 Best Model: {best_model}")
        logger.info(f"   Accuracy: {comparison.loc[best_model, 'accuracy']:.4f}")
        logger.info(f"   ROC-AUC: {comparison.loc[best_model, 'roc_auc']:.4f}")
        
        return comparison
    
    def save_models(self, directory: str = "models"):
        """
        Save all trained models and preprocessing objects.
        
        Args:
            directory: Directory to save models
        """
        Path(directory).mkdir(exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            filename = Path(directory) / f"{name.lower().replace(' ', '_')}.pkl"
            joblib.dump(model, filename)
            logger.info(f"Saved {name} to {filename}")
        
        # Save scaler
        joblib.dump(self.scaler, Path(directory) / "scaler.pkl")
        logger.info(f"Saved scaler to {Path(directory) / 'scaler.pkl'}")
        
        # Save label encoders
        joblib.dump(self.label_encoders, Path(directory) / "label_encoders.pkl")
        logger.info(f"Saved label encoders to {Path(directory) / 'label_encoders.pkl'}")
        
        # Save feature columns
        joblib.dump(self.feature_columns, Path(directory) / "feature_columns.pkl")
        logger.info(f"Saved feature columns to {Path(directory) / 'feature_columns.pkl'}")
        
        # Save results
        comparison = self.compare_models()
        comparison.to_csv(Path(directory) / "model_comparison.csv")
        
        logger.info(f"\n✅ All models and artifacts saved to {directory}/")
    
    def run_pipeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            df: Preprocessed and engineered DataFrame
        
        Returns:
            Dictionary with results and trained models
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("=" * 60)
        
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        self.split_data(X, y)
        
        # Train models
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_voting_ensemble()
        
        # Compare and save
        comparison = self.compare_models()
        self.save_models()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ MODEL TRAINING COMPLETE!")
        logger.info("=" * 60)
        
        return {
            'results': self.results,
            'comparison': comparison,
            'models': self.models,
            'best_model': comparison['accuracy'].idxmax()
        }


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Load and prepare data (you'll need actual data for real testing)
    print("\n🧪 Testing ModelTrainer...")
    print("✅ ModelTrainer class ready to use!")
    print("\nNote: Run with actual Titanic data for full training")