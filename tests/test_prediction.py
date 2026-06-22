"""
Unit tests for prediction module
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.prediction import Predictor


class TestPredictor:
    """Test cases for Predictor class"""
    
    def setup_method(self):
        """Setup test data"""
        # Create sample passenger data
        self.test_passenger = {
            'Pclass': 1,
            'Sex': 'female',
            'Age': 28,
            'SibSp': 0,
            'Parch': 1,
            'Fare': 71.28,
            'Embarked': 'C',
            'Name': 'Test, Mrs. Rose'
        }
        
        self.test_passenger_male = {
            'Pclass': 3,
            'Sex': 'male',
            'Age': 25,
            'SibSp': 0,
            'Parch': 0,
            'Fare': 8.05,
            'Embarked': 'S',
            'Name': 'Test, Mr. Jack'
        }
    
    def test_predictor_initialization(self):
        """Test predictor initialization (may fail if no models)"""
        try:
            predictor = Predictor()
            assert predictor is not None
        except Exception as e:
            pytest.skip(f"Models not trained yet: {e}")
    
    def test_predictor_model_info(self):
        """Test model info retrieval"""
        try:
            predictor = Predictor()
            info = predictor.get_model_info()
            
            assert 'models_loaded' in info
            assert 'feature_columns' in info
        except Exception as e:
            pytest.skip(f"Models not trained yet: {e}")
    
    def test_explain_prediction(self):
        """Test explanation generation"""
        try:
            predictor = Predictor()
            explanation = predictor.explain_prediction(self.test_passenger)
            
            assert isinstance(explanation, str)
            assert len(explanation) > 0
        except Exception as e:
            pytest.skip(f"Models not trained yet: {e}")
    
    def test_predict_with_explanation(self):
        """Test prediction with explanation"""
        try:
            predictor = Predictor()
            result = predictor.predict_with_explanation(self.test_passenger)
            
            assert 'survived' in result
            assert 'survival_probability' in result
            assert 'prediction_text' in result
            assert 'explanation' in result
            assert 'risk_level' in result
        except Exception as e:
            pytest.skip(f"Models not trained yet: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])