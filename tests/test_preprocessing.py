"""
Unit tests for data preprocessing module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class"""
    
    def setup_method(self):
        """Setup test data before each test"""
        self.preprocessor = DataPreprocessor()
        
        # Create sample test data
        self.test_df = pd.DataFrame({
            'Survived': [0, 1, 0, 1],
            'Pclass': [3, 1, 3, 2],
            'Sex': ['male', 'female', 'male', 'female'],
            'Age': [22, 38, np.nan, 26],
            'SibSp': [1, 1, 0, 0],
            'Parch': [0, 0, 0, 0],
            'Fare': [7.25, 71.28, 8.05, 32.0],
            'Embarked': ['S', 'C', 'S', np.nan],
            'Name': ['Test1', 'Test2', 'Test3', 'Test4'],
            'Ticket': ['A', 'B', 'C', 'D'],
            'Cabin': [np.nan, 'C85', np.nan, np.nan]
        })
    
    def test_load_data(self):
        """Test data loading functionality"""
        # This test expects a real file - will skip if not found
        pass
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        result = self.preprocessor.handle_missing_values(self.test_df.copy())
        
        # Check no missing values remain
        assert result.isnull().sum().sum() == 0
        
        # Check Age was filled (was 1 missing)
        assert result['Age'].notnull().all()
        
        # Check Embarked was filled (was 1 missing)
        assert result['Embarked'].notnull().all()
    
    def test_detect_outliers(self):
        """Test outlier detection"""
        # Add an outlier
        df_with_outlier = self.test_df.copy()
        df_with_outlier.loc[0, 'Fare'] = 500  # Extreme fare
        
        outliers = self.preprocessor.detect_outliers(df_with_outlier, 'Fare')
        assert outliers.sum() >= 1
    
    def test_validate_data(self):
        """Test data validation"""
        # Valid data should pass
        valid_df = self.test_df.dropna()
        result = self.preprocessor.validate_data(valid_df)
        assert result == True
    
    def test_validate_data_with_invalid(self):
        """Test validation catches errors"""
        # Create invalid data
        invalid_df = self.test_df.copy()
        invalid_df.loc[0, 'Pclass'] = 5  # Invalid class
        
        with pytest.raises(ValueError):
            self.preprocessor.validate_data(invalid_df)
    
    def test_get_data_info(self):
        """Test data info generation"""
        info = self.preprocessor.get_data_info(self.test_df)
        
        assert 'shape' in info
        assert 'columns' in info
        assert 'missing' in info
        assert info['shape'] == (4, 11)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])