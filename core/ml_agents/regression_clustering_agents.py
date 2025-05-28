from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import KMeans
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from .base_agent import MLAgent

class LinearRegressionAgent(MLAgent):
    """Specialized agent for linear regression tasks"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("LinearRegression", model_path)
        self.model = LinearRegression(n_jobs=-1)
    
    def train(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Train the linear regression model"""
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Get feature importance (coefficients)
        feature_importance = dict(zip(X.columns, np.abs(self.model.coef_)))
        
        # Save model
        self.save_model()
        
        return {
            'feature_importance': feature_importance,
            'model_params': self.model.get_params()
        }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(data)
    
    def analyze(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze model performance"""
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        y_pred = self.predict(X)
        
        return {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred)
        }

class DecisionTreeAgent(MLAgent):
    """Specialized agent for decision tree tasks"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("DecisionTree", model_path)
        self.model = DecisionTreeRegressor(
            max_depth=None,
            random_state=42
        )
    
    def train(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Train the decision tree model"""
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Get feature importance
        feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        # Save model
        self.save_model()
        
        return {
            'feature_importance': feature_importance,
            'model_params': self.model.get_params()
        }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(data)
    
    def analyze(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze model performance"""
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        y_pred = self.predict(X)
        
        return {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred)
        }

class KMeansAgent(MLAgent):
    """Specialized agent for K-means clustering tasks"""
    
    def __init__(self, model_path: Optional[str] = None, n_clusters: int = 5):
        super().__init__("KMeans", model_path)
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the K-means clustering model"""
        self.model.fit(data)
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return {
            'n_clusters': self.model.n_clusters,
            'inertia': float(self.model.inertia_),
            'model_params': self.model.get_params()
        }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Assign clusters to data points"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(data)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze clustering performance"""
        labels = self.predict(data)
        
        return {
            'n_clusters': self.model.n_clusters,
            'silhouette': float(silhouette_score(data, labels)),
            'calinski_harabasz': float(calinski_harabasz_score(data, labels)),
            'davies_bouldin': float(davies_bouldin_score(data, labels)),
            'cluster_sizes': dict(zip(
                range(self.model.n_clusters),
                np.bincount(labels)
            ))
        }

class LightGBMAgent(MLAgent):
    """Specialized agent for LightGBM tasks"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("LightGBM", model_path)
        self.model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=-1,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Train the LightGBM model"""
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Get feature importance
        feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        # Save model
        self.save_model()
        
        return {
            'feature_importance': feature_importance,
            'model_params': self.model.get_params()
        }
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(data)
    
    def analyze(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze model performance"""
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        y_pred = self.predict(X)
        
        return {
            'r2': r2_score(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred)
        } 