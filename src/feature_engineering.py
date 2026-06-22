"""
Feature Engineering Module
==========================
Creates new features from existing data to improve model performance.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from .config import config

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for Titanic dataset.
    
    Creates features:
        - FamilySize: Number of family members aboard
        - IsAlone: Whether passenger traveled alone
        - Title: Social title extracted from name
        - AgeGroup: Binned age categories
        - FareGroup: Binned fare categories
    """
    
    def __init__(self):
        """Initialize feature engineer with mappings"""
        # Title mapping for grouping rare titles
        self.title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        
        # Age group bins and labels
        self.age_bins = [0, 12, 18, 35, 60, 100]
        self.age_labels = ['Child', 'Teenager', 'Adult', 'MiddleAge', 'Senior']
        
        # Fare group bins (fixed bins instead of quantiles, safe for single rows)
        self.fare_bins = [0, 7.91, 14.45, 31.0, 99.0, float('inf')]
        self.fare_labels = ['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh']
    
    def create_family_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create family-related features.
        
        Features:
            - FamilySize: Total family members aboard (self + siblings + parents)
            - IsAlone: 1 if traveling alone, 0 otherwise
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with added family features
        """
        logger.info("Creating family features...")
        
        # Family size (self + siblings + spouse + parents + children)
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        
        # Is alone (no family members)
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
        
        logger.info(f"  - FamilySize range: {df['FamilySize'].min()} to {df['FamilySize'].max()}")
        logger.info(f"  - IsAlone: {df['IsAlone'].sum()} passengers alone ({df['IsAlone'].mean()*100:.1f}%)")
        
        return df
    
    def create_title_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract social title from passenger names.
        Falls back to 'Mr'/'Mrs' based on Sex if Name is missing.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with added Title feature
        """
        logger.info("Creating title feature...")
        
        if 'Name' in df.columns and df['Name'].notna().any():
            # Extract title using regex (looks for word followed by a dot)
            df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
            
            # Map to grouped categories
            df['Title'] = df['Title'].map(self.title_mapping)
        else:
            df['Title'] = None

        # Fill missing titles based on Sex as fallback
        if 'Sex' in df.columns:
            mask = df['Title'].isna()
            df.loc[mask, 'Title'] = df.loc[mask, 'Sex'].map(
                {'male': 'Mr', 'female': 'Mrs'}
            )

        # Final fallback
        df['Title'] = df['Title'].fillna('Rare')
        
        logger.info(f"  - Titles assigned: {df['Title'].value_counts().to_dict()}")
        
        return df
    
    def create_age_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age group categories.
        
        Age groups:
            - Child: 0-12 years
            - Teenager: 12-18 years
            - Adult: 18-35 years
            - MiddleAge: 35-60 years
            - Senior: 60+ years
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with added AgeGroup feature
        """
        logger.info("Creating age groups...")
        
        df['AgeGroup'] = pd.cut(
            df['Age'], 
            bins=self.age_bins, 
            labels=self.age_labels, 
            right=False
        )
        
        # Fill any NaN age groups with 'Adult' as default
        if hasattr(df['AgeGroup'], 'cat') and 'Adult' not in df['AgeGroup'].cat.categories:
            df['AgeGroup'] = df['AgeGroup'].cat.add_categories(['Adult'])
        df['AgeGroup'] = df['AgeGroup'].fillna('Adult')
        
        logger.info(f"  - AgeGroup distribution: {df['AgeGroup'].value_counts().to_dict()}")
        
        return df
    
    def create_fare_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create fare group categories using fixed bins.
        Uses fixed bins instead of quantiles so it works for single rows too.
        
        Groups:
            - VeryLow:  $0 - $7.91
            - Low:      $7.91 - $14.45
            - Medium:   $14.45 - $31.00
            - High:     $31.00 - $99.00
            - VeryHigh: $99.00+
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with added FareGroup feature
        """
        logger.info("Creating fare groups...")
        
        df['FareGroup'] = pd.cut(
            df['Fare'],
            bins=self.fare_bins,
            labels=self.fare_labels,
            right=True,
            include_lowest=True
        )
        
        # Fill any NaN fare groups with 'Medium' as default
        df['FareGroup'] = df['FareGroup'].fillna('Medium')
        
        logger.info(f"  - FareGroup distribution: {df['FareGroup'].value_counts().to_dict()}")
        
        return df
    
    def create_deck_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract deck letter from cabin number (if available).
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with added Deck feature
        """
        if 'Cabin' in df.columns:
            logger.info("Creating deck feature from cabin...")
            df['Deck'] = df['Cabin'].str[0]
            df['Deck'] = df['Deck'].fillna('Unknown')
            logger.info(f"  - Decks found: {df['Deck'].value_counts().head(5).to_dict()}")
        else:
            logger.info("  - Cabin column not available, skipping deck feature")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run all feature engineering steps.
        Safe to call on a single-row DataFrame (e.g. during prediction).
        
        Args:
            df: Input DataFrame (should be preprocessed)
        
        Returns:
            DataFrame with all engineered features added
        """
        logger.info("=" * 50)
        logger.info("Starting feature engineering...")
        logger.info("=" * 50)
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Drop Survived if accidentally passed in
        if 'Survived' in df.columns:
            df = df.drop(columns=['Survived'])
        
        # Create all features
        df = self.create_family_features(df)
        df = self.create_title_feature(df)
        df = self.create_age_groups(df)
        df = self.create_fare_groups(df)
        df = self.create_deck_feature(df)
        
        logger.info("=" * 50)
        logger.info("✅ Feature engineering complete!")
        logger.info(f"   Total features now: {len(df.columns)}")
        logger.info("=" * 50)
        
        return df
    
    def get_feature_columns(self, include_target: bool = False) -> List[str]:
        """
        Get list of feature columns for modeling.
        
        Args:
            include_target: Whether to include target column
        
        Returns:
            List of feature column names
        """
        numerical = config.features.get('numerical', [])
        categorical = config.features.get('categorical', [])
        engineered = config.features.get('engineered', [])
        
        features = numerical + categorical + engineered
        
        encoded_features = [
            'Sex_Encoded',
            'Embarked_C', 'Embarked_Q', 'Embarked_S',
            'Title_Mr', 'Title_Mrs', 'Title_Miss', 'Title_Master', 'Title_Rare',
            'AgeGroup_Child', 'AgeGroup_Teenager', 'AgeGroup_Adult', 
            'AgeGroup_MiddleAge', 'AgeGroup_Senior',
            'FareGroup_VeryLow', 'FareGroup_Low', 'FareGroup_Medium',
            'FareGroup_High', 'FareGroup_VeryHigh'
        ]
        
        all_features = features + encoded_features
        
        if not include_target and 'Survived' in all_features:
            all_features.remove('Survived')
        
        return all_features


# For testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with a single row (prediction scenario)
    test_single = pd.DataFrame({
        'Name': ['Braund, Mr. Owen Harris'],
        'Age': [22],
        'SibSp': [1],
        'Parch': [0],
        'Fare': [7.25],
        'Sex': ['male'],
        'Embarked': ['S']
    })
    
    engineer = FeatureEngineer()
    print("\n🧪 Testing FeatureEngineer with single row...")
    result = engineer.engineer_features(test_single)
    print(f"New features: {[c for c in result.columns if c not in test_single.columns]}")
    print("✅ FeatureEngineer class ready to use!")