#!/usr/bin/env python
"""
Titanic Survival Prediction - Main Pipeline
============================================
Complete pipeline for data preprocessing, feature engineering,
model training, and evaluation.

Usage:
    python run_pipeline.py          # Run full pipeline
    python run_pipeline.py --train  # Train models only
    python run_pipeline.py --eval   # Evaluate only
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure Unicode output in environments with limited console encoding
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Project root and import path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from src.config import config
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.visualization import Visualizer
from src.prediction import Predictor

# Ensure log directories exist before logging to files
for file_path in config.logging.get('files', {}).values():
    log_path = Path(file_path)
    if not log_path.is_absolute():
        log_path = PROJECT_ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.logging.get('level', 'INFO')),
    format=config.logging.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    handlers=[
        logging.FileHandler(config.logging.get('files', {}).get('data_pipeline', 'logs/data_pipeline.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'models', 'logs', 'reports/figures']
    for d in directories:
        path = PROJECT_ROOT / d
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {path.relative_to(PROJECT_ROOT)}")


def load_and_preprocess_data():
    """Load and preprocess the Titanic dataset"""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Loading and Preprocessing Data")
    logger.info("="*60)
    
    preprocessor = DataPreprocessor()
    
    data_path = Path(config.data.get('raw_path', 'data/titanic.csv'))
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    df = preprocessor.load_data(str(data_path))
    
    logger.info(f"\nDataset Info:")
    logger.info(f"  - Shape: {df.shape}")
    logger.info(f"  - Columns: {list(df.columns)}")
    logger.info(f"  - Missing values: {df.isnull().sum().sum()}")
    
    # Pass include_target=True since this is training data
    df_clean = preprocessor.preprocess(df, include_target=True)
    
    logger.info(f"\nAfter preprocessing:")
    logger.info(f"  - Shape: {df_clean.shape}")
    logger.info(f"  - Missing values: {df_clean.isnull().sum().sum()}")
    
    return df_clean


def engineer_features(df):
    """Create engineered features — keeps Survived column intact"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Feature Engineering")
    logger.info("="*60)
    
    engineer = FeatureEngineer()

    # Temporarily remove Survived so engineer doesn't drop it,
    # then add it back after engineering
    survived = df['Survived'].copy() if 'Survived' in df.columns else None
    df_no_target = df.drop(columns=['Survived'], errors='ignore')

    df_features = engineer.engineer_features(df_no_target)

    # Re-attach Survived for model training
    if survived is not None:
        df_features['Survived'] = survived.values

    logger.info(f"\nAfter feature engineering:")
    logger.info(f"  - Shape: {df_features.shape}")
    logger.info(f"  - New features: {[c for c in df_features.columns if c not in df.columns]}")
    
    return df_features


def create_visualizations(df):
    """Create EDA visualizations — pass df with Survived column"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Creating Visualizations")
    logger.info("="*60)
    
    visualizer = Visualizer()
    visualizer.create_eda_report(df)
    
    return visualizer


def train_models(df):
    """Train and evaluate all models"""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Training Models")
    logger.info("="*60)
    
    trainer = ModelTrainer()
    results = trainer.run_pipeline(df)
    
    return trainer, results


def test_prediction(trainer):
    """Test prediction with sample data"""
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Testing Predictions")
    logger.info("="*60)
    
    test_passengers = [
        {
            'Pclass': 1,
            'Sex': 'female',
            'Age': 28,
            'SibSp': 0,
            'Parch': 1,
            'Fare': 71.28,
            'Embarked': 'C',
            'Name': 'Test, Mrs. Rose'
        },
        {
            'Pclass': 3,
            'Sex': 'male',
            'Age': 25,
            'SibSp': 0,
            'Parch': 0,
            'Fare': 8.05,
            'Embarked': 'S',
            'Name': 'Test, Mr. Jack'
        },
        {
            'Pclass': 2,
            'Sex': 'female',
            'Age': 8,
            'SibSp': 1,
            'Parch': 2,
            'Fare': 29.0,
            'Embarked': 'S',
            'Name': 'Test, Miss. Young'
        }
    ]
    
    try:
        predictor = Predictor()
        
        for i, passenger in enumerate(test_passengers, 1):
            logger.info(f"\nTest Passenger {i}:")
            logger.info(f"  - Class: {passenger['Pclass']}, Gender: {passenger['Sex']}, Age: {passenger['Age']}")
            
            result = predictor.predict_with_explanation(passenger)
            logger.info(f"  - Prediction: {result['prediction_text']}")
            logger.info(f"  - Probability: {result['survival_probability']:.1%}")
            logger.info(f"  - Risk: {result['risk_level']}")
            
    except Exception as e:
        logger.warning(f"Prediction test skipped: {e}")
        logger.info("Run training first to save models")


def generate_report(trainer, visualizer, df):
    """Generate final report — expects df with Survived column"""
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Generating Final Report")
    logger.info("="*60)
    
    comparison = trainer.compare_models()
    
    report_path = PROJECT_ROOT / 'reports' / 'model_comparison.csv'
    comparison.to_csv(report_path)
    logger.info(f"Model comparison saved to {report_path.relative_to(PROJECT_ROOT)}")
    
    best_model = comparison['accuracy'].idxmax()
    best_accuracy = comparison.loc[best_model, 'accuracy']
    
    logger.info(f"\nDataset: {df.shape[0]} passengers, {df.shape[1]} features")
    logger.info(f"Best Model: {best_model}")
    logger.info(f"Best Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    logger.info("\nAll Model Performances:")
    for model in comparison.index:
        logger.info(f"  - {model}: {comparison.loc[model, 'accuracy']:.4f} accuracy, {comparison.loc[model, 'roc_auc']:.4f} ROC-AUC")
    
    if 'Survived' in df.columns:
        survival_rate = df['Survived'].mean() * 100
        logger.info(f"\nHistorical Survival Rate: {survival_rate:.1f}%")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Titanic Survival Prediction Pipeline')
    parser.add_argument('--train', action='store_true', help='Train models only')
    parser.add_argument('--eval', action='store_true', help='Evaluate only')
    parser.add_argument('--skip-viz', action='store_true', help='Skip visualizations')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("TITANIC SURVIVAL PREDICTION PIPELINE")
    logger.info("="*60)
    logger.info(f"Project: {config.project.get('name')}")
    logger.info(f"Version: {config.project.get('version')}")
    logger.info("="*60)
    
    setup_directories()
    
    try:
        # Step 1: Load and preprocess (df_clean has Survived)
        df_clean = load_and_preprocess_data()
        
        # Step 2: Engineer features (df_features also has Survived re-attached)
        df_features = engineer_features(df_clean)
        
        # Step 3: Visualizations — use df_clean which has Survived
        if not args.skip_viz:
            visualizer = create_visualizations(df_clean)
        else:
            visualizer = None
        
        # Step 4: Train models — df_features has Survived for training
        if not args.eval:
            trainer, results = train_models(df_features)
            
            # Step 5: Test predictions
            test_prediction(trainer)
            
            # Step 6: Report — use df_clean which has Survived
            generate_report(trainer, visualizer, df_clean)
        else:
            logger.info("\nEvaluation mode - skipping training")
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info("\nTo launch the GUI, run:")
        logger.info("   streamlit run gui/app.py")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()