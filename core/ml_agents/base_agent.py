from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from sklearn.base import BaseEstimator
import joblib
import os
from pathlib import Path

class MLAgent(ABC):
    """Base class for all ML agents"""
    
    def __init__(self, name: str, model_path: Optional[str] = None):
        self.name = name
        self.model_path = model_path
        self.model: Optional[BaseEstimator] = None
        self.is_trained = False
        self.metadata: Dict[str, Any] = {}
        
        # Create model directory if it doesn't exist
        if model_path:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    @abstractmethod
    def train(self, data: pd.DataFrame) -> None:
        """Train the agent's model"""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Any:
        """Make predictions using the trained model"""
        pass
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data and return insights"""
        pass
    
    def save_model(self) -> None:
        """Save the trained model to disk"""
        if self.model and self.model_path:
            joblib.dump(self.model, self.model_path)
            print(f"✅ Model saved to {self.model_path}")
    
    def load_model(self) -> None:
        """Load a trained model from disk"""
        if self.model_path and os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            self.is_trained = True
            print(f"✅ Model loaded from {self.model_path}")
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update agent metadata"""
        self.metadata[key] = value
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get agent metadata"""
        return self.metadata
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if data.empty:
            raise ValueError("DataFrame is empty")
        return True
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Basic data preprocessing"""
        # Remove duplicate rows
        data = data.drop_duplicates()
        
        # Handle missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            data[col] = data[col].fillna(data[col].median())
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            data[col] = data[col].fillna(data[col].mode()[0])
        
        return data
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        if not self.model or not hasattr(self.model, 'feature_importances_'):
            return None
        
        return {
            f"feature_{i}": importance 
            for i, importance in enumerate(self.model.feature_importances_)
        }
    
    def __str__(self) -> str:
        return f"{self.name} Agent (Trained: {self.is_trained})"
    
    def __repr__(self) -> str:
        return self.__str__() 