"""
Configuration Management Module
===============================
Loads and manages project configuration from YAML file.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Optional


class Config:
    """
    Singleton configuration manager for the entire project.
    
    Usage:
        config = Config()
        data_path = config.data['raw_path']
        random_state = config.get('data.random_state', 42)
    """
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern - only one instance exists"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from YAML file"""
        # Look for config.yaml in parent directory (project root)
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {config_path}\n"
                "Please ensure config.yaml exists in the project root directory."
            )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    @property
    def project(self) -> dict:
        """Project information"""
        return self._config.get('project', {})
    
    @property
    def data(self) -> dict:
        """Data configuration"""
        return self._config.get('data', {})
    
    @property
    def preprocessing(self) -> dict:
        """Preprocessing configuration"""
        return self._config.get('preprocessing', {})
    
    @property
    def features(self) -> dict:
        """Feature configuration"""
        return self._config.get('features', {})
    
    @property
    def models(self) -> dict:
        """Model configuration"""
        return self._config.get('models', {})
    
    @property
    def evaluation(self) -> dict:
        """Evaluation configuration"""
        return self._config.get('evaluation', {})
    
    @property
    def gui(self) -> dict:
        """GUI configuration"""
        return self._config.get('gui', {})
    
    @property
    def logging(self) -> dict:
        """Logging configuration"""
        return self._config.get('logging', {})
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example:
            config.get('data.random_state', 42)
            config.get('models.random_forest.n_estimators')
        
        Args:
            key: Dot-separated path to configuration value
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __repr__(self) -> str:
        return f"Config({list(self._config.keys())})"


# Create global config instance for easy import
config = Config()


# Quick test if run directly
if __name__ == "__main__":
    print("Testing configuration loading...")
    print(f"Project name: {config.project.get('name')}")
    print(f"Data path: {config.data.get('raw_path')}")
    print(f"Random state: {config.get('data.random_state')}")
    print("✅ Configuration loaded successfully!")