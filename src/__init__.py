"""
Titanic Survival Prediction Package
===================================
A complete machine learning project for predicting Titanic passenger survival.

Author: Group 2 - IIUI
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Group 2 - IIUI"
__all__ = [
    'DataPreprocessor',
    'FeatureEngineer', 
    'ModelTrainer',
    'Predictor',
    'Visualizer'
]

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .prediction import Predictor
from .visualization import Visualizer