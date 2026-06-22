"""
Unit tests for feature engineering module
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class"""
    
    def setup_method(self):
        """Setup test data before each test"""
        self.engineer = FeatureEngineer()
        
        self.test_df = pd.DataFrame({
            'Name': ['Braund, Mr. Owen Harris', 'Cumings, Mrs. John', 'Heikkinen, Miss. Laina'],
            'Age': [22, 38, 26],
            'SibSp': [1, 1, 0],
            'Parch': [0, 0, 0],
            'Fare': [7.25, 71.28, 7.92],
            'Sex': ['male', 'female', 'female'],
            'Embarked': ['S', 'C', 'S']
        })
    
    def test_create_family_features(self):
        """Test family feature creation"""
        result = self.engineer.create_family_features(self.test_df.copy())
        
        assert 'FamilySize' in result.columns
        assert 'IsAlone' in result.columns
        
        # Test calculations
        assert result.loc[0, 'FamilySize'] == 2  # 1 SibSp + 0 Parch + 1
        assert result.loc[2, 'FamilySize'] == 1  # Alone
        assert result.loc[2, 'IsAlone'] == 1      # Is alone
    
    def test_create_title_feature(self):
        """Test title extraction"""
        result = self.engineer.create_title_feature(self.test_df.copy())
        
        assert 'Title' in result.columns
        assert result.loc[0, 'Title'] == 'Mr'
        assert result.loc[1, 'Title'] == 'Mrs'
        assert result.loc[2, 'Title'] == 'Miss'
    
    def test_create_age_groups(self):
        """Test age group creation"""
        result = self.engineer.create_age_groups(self.test_df.copy())
        
        assert 'AgeGroup' in result.columns
        
        # Test age group assignment
        # Age 22 should be 'Adult' (18-35)
        assert result.loc[0, 'AgeGroup'] == 'Adult'
    
    def test_create_fare_groups(self):
        """Test fare group creation"""
        result = self.engineer.create_fare_groups(self.test_df.copy())
        
        assert 'FareGroup' in result.columns
    
    def test_engineer_features(self):
        """Test complete feature engineering pipeline"""
        result = self.engineer.engineer_features(self.test_df.copy())
        
        # All new features should be present
        expected_features = ['FamilySize', 'IsAlone', 'Title', 'AgeGroup', 'FareGroup']
        
        for feature in expected_features:
            assert feature in result.columns
    
    def test_get_feature_columns(self):
        """Test feature column list generation"""
        features = self.engineer.get_feature_columns()
        
        assert isinstance(features, list)
        assert len(features) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])