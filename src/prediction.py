"""
Prediction Module
=================
Makes predictions using trained models for new passenger data.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .config import config
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class Predictor:
    """
    Prediction class for making survival predictions on new passengers.
    
    Handles:
        - Loading trained models
        - Preprocessing new input data
        - Making predictions
        - Returning probabilities and explanations
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize predictor with trained models.
        
        Args:
            models_dir: Directory containing saved models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.label_encoders = None
        self.feature_columns = None
        self.preprocessor = DataPreprocessor()
        self.engineer = FeatureEngineer()
        
        # Load all models and artifacts
        self._load_models()
    
    def _load_models(self):
        """Load all trained models and preprocessing artifacts"""
        logger.info(f"Loading models from {self.models_dir}")
        
        try:
            # Load models
            model_files = {
                'logistic_regression': 'logistic_regression.pkl',
                'random_forest': 'random_forest.pkl',
                'voting_ensemble': 'voting_ensemble.pkl'
            }
            
            for name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[name] = joblib.load(model_path)
                    logger.info(f"  - Loaded {name}")
                else:
                    logger.warning(f"  - {name} model not found at {model_path}")
            
            # Load scaler
            scaler_path = self.models_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("  - Loaded scaler")
            
            # Load label encoders
            encoders_path = self.models_dir / "label_encoders.pkl"
            if encoders_path.exists():
                self.label_encoders = joblib.load(encoders_path)
                logger.info("  - Loaded label encoders")
            
            # Load feature columns
            features_path = self.models_dir / "feature_columns.pkl"
            if features_path.exists():
                self.feature_columns = joblib.load(features_path)
                logger.info(f"  - Loaded {len(self.feature_columns)} feature columns")
            
            logger.info("✅ Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _simple_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simple preprocessing for prediction (doesn't require Survived column)
        
        Args:
            df: Input DataFrame
        
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Fill missing Age with median
        if 'Age' in df.columns:
            age_median = 28  # Default Titanic age median
            if hasattr(self.preprocessor, 'age_median') and self.preprocessor.age_median:
                age_median = self.preprocessor.age_median
            df['Age'] = df['Age'].fillna(age_median)
        
        # Fill missing Embarked with mode
        if 'Embarked' in df.columns:
            embarked_mode = 'S'  # Default mode
            if hasattr(self.preprocessor, 'embarked_mode') and self.preprocessor.embarked_mode:
                embarked_mode = self.preprocessor.embarked_mode
            df['Embarked'] = df['Embarked'].fillna(embarked_mode)
        
        # Fill missing Fare with mean
        if 'Fare' in df.columns:
            fare_mean = 32.2  # Default Titanic fare mean
            df['Fare'] = df['Fare'].fillna(fare_mean)
        
        return df
    
    def _prepare_input(self, passenger_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare single passenger input for prediction.
        
        Args:
            passenger_data: Dictionary with passenger features
        
        Returns:
            DataFrame ready for model prediction
        """
        # Convert to DataFrame
        df = pd.DataFrame([passenger_data])
        
        # Apply simple preprocessing (no Survived column needed)
        # THIS IS THE KEY FIX - NOT calling preprocessor.preprocess()
        df = self._simple_preprocess(df)
        
        # Apply feature engineering
        df = self.engineer.engineer_features(df)
        
        # Create feature matrix
        X = pd.DataFrame(index=df.index)
        
        # Add basic features
        for col in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
            if col in df.columns:
                X[col] = df[col]
        
        # Add family features
        if 'FamilySize' in df.columns:
            X['FamilySize'] = df['FamilySize']
        if 'IsAlone' in df.columns:
            X['IsAlone'] = df['IsAlone']
        
        # Encode Sex
        if 'Sex' in df.columns and self.label_encoders and 'Sex' in self.label_encoders:
            X['Sex_Encoded'] = self.label_encoders['Sex'].transform(df['Sex'])
        
        # One-hot encode Embarked
        if 'Embarked' in df.columns:
            embarked = df['Embarked'].iloc[0]
            for port in ['C', 'Q', 'S']:
                col_name = f'Embarked_{port}'
                X[col_name] = 1 if embarked == port else 0
        
        # Ensure all feature columns are present
        if self.feature_columns is not None:
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            
            # Reorder columns to match training
            X = X[self.feature_columns]
        
        # Scale numerical features
        numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        numerical_present = [col for col in numerical_features if col in X.columns]
        
        if numerical_present and self.scaler is not None:
            X[numerical_present] = self.scaler.transform(X[numerical_present])
        
        return X
    
    def predict(self, passenger_data: Dict[str, Any], model_name: str = "random_forest") -> Dict[str, Any]:
        """
        Make prediction for a single passenger.
        
        Args:
            passenger_data: Dictionary containing passenger information
            model_name: Which model to use ('logistic_regression', 'random_forest', 'voting_ensemble')
        
        Returns:
            Dictionary with prediction results
        """
        logger.info(f"Making prediction using {model_name}")
        
        # Prepare input
        X = self._prepare_input(passenger_data)
        
        # Get model
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Create result
        result = {
            'survived': bool(prediction),
            'survival_probability': float(probability[1]),
            'death_probability': float(probability[0]),
            'model_used': model_name,
            'prediction_text': "SURVIVED" if prediction == 1 else "DID NOT SURVIVE",
            'confidence': float(max(probability)) * 100
        }
        
        # Add risk assessment
        if result['survival_probability'] >= 0.7:
            result['risk_level'] = "Low Risk (Good chances)"
            result['emoji'] = "✅"
        elif result['survival_probability'] >= 0.4:
            result['risk_level'] = "Medium Risk (Uncertain)"
            result['emoji'] = "⚠️"
        else:
            result['risk_level'] = "High Risk (Low chances)"
            result['emoji'] = "❌"
        
        logger.info(f"Prediction: {result['prediction_text']} (Probability: {result['survival_probability']:.1%})")
        
        return result
    
    def predict_batch(self, passengers: list, model_name: str = "random_forest") -> pd.DataFrame:
        """
        Make predictions for multiple passengers.
        
        Args:
            passengers: List of passenger dictionaries
            model_name: Which model to use
        
        Returns:
            DataFrame with predictions for all passengers
        """
        results = []
        for i, passenger in enumerate(passengers):
            try:
                result = self.predict(passenger, model_name)
                result['passenger_id'] = i + 1
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for passenger {i+1}: {e}")
                results.append({
                    'passenger_id': i + 1,
                    'error': str(e),
                    'survived': None
                })
        
        return pd.DataFrame(results)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'models_loaded': list(self.models.keys()),
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns) if self.feature_columns else 0,
            'scaler_loaded': self.scaler is not None,
            'encoders_loaded': self.label_encoders is not None
        }
        return info
    
    def explain_prediction(self, passenger_data: Dict[str, Any]) -> str:
        """
        Generate human-readable explanation of the prediction.
        
        Args:
            passenger_data: Passenger information
        
        Returns:
            String explaining the prediction
        """
        explanations = []
        
        # Gender impact
        if passenger_data.get('Sex') == 'female':
            explanations.append("• Women had a 74% survival rate on the Titanic (much higher than men)")
        else:
            explanations.append("• Men had only a 19% survival rate on the Titanic")
        
        # Class impact
        pclass = passenger_data.get('Pclass', 0)
        if pclass == 1:
            explanations.append("• First class passengers had the highest survival rate (63%)")
        elif pclass == 2:
            explanations.append("• Second class passengers had a moderate survival rate (47%)")
        elif pclass == 3:
            explanations.append("• Third class passengers had the lowest survival rate (24%)")
        
        # Age impact
        age = passenger_data.get('Age', 30)
        if age < 12:
            explanations.append("• Children had better survival chances")
        elif age > 60:
            explanations.append("• Elderly passengers faced additional challenges during evacuation")
        
        # Family impact
        sibsp = passenger_data.get('SibSp', 0)
        parch = passenger_data.get('Parch', 0)
        family_size = sibsp + parch + 1
        
        if family_size == 1:
            explanations.append("• Traveling alone made evacuation more difficult")
        elif 2 <= family_size <= 4:
            explanations.append("• Small families had better coordination during evacuation")
        elif family_size > 4:
            explanations.append("• Large families faced challenges staying together")
        
        return "\n".join(explanations)
    
    def predict_with_explanation(self, passenger_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction with detailed explanation.
        
        Args:
            passenger_data: Passenger information
        
        Returns:
            Dictionary with prediction and explanation
        """
        # Get prediction
        result = self.predict(passenger_data)
        
        # Add explanation
        result['explanation'] = self.explain_prediction(passenger_data)
        
        # Add feature summary
        result['features'] = {
            'Gender': passenger_data.get('Sex', 'Unknown'),
            'Class': passenger_data.get('Pclass', 'Unknown'),
            'Age': passenger_data.get('Age', 'Unknown'),
            'Fare': f"${passenger_data.get('Fare', 0):.2f}",
            'Family Size': passenger_data.get('SibSp', 0) + passenger_data.get('Parch', 0) + 1
        }
        
        return result


# Create a singleton instance for easy import
_predictor_instance = None

def get_predictor() -> Predictor:
    """Get or create predictor singleton instance"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = Predictor()
    return _predictor_instance


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test prediction (will work after models are trained)
    print("\n🧪 Testing Predictor...")
    
    test_passenger = {
        'Pclass': 1,
        'Sex': 'female',
        'Age': 28,
        'SibSp': 0,
        'Parch': 1,
        'Fare': 71.28,
        'Embarked': 'C',
        'Name': 'Test, Mrs.'
    }
    
    try:
        predictor = Predictor()
        result = predictor.predict_with_explanation(test_passenger)
        print(f"\nPrediction Result:")
        print(f"  - Survived: {result['survived']}")
        print(f"  - Probability: {result['survival_probability']:.1%}")
        print(f"  - Risk: {result['risk_level']}")
        print(f"\nExplanation:\n{result['explanation']}")
    except Exception as e:
        print(f"Note: Train models first using run_pipeline.py")
        print(f"Error: {e}")
    
    print("\n✅ Predictor class ready to use!")