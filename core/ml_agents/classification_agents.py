from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .base_agent import MLAgent

class LogisticRegressionAgent(MLAgent):
    """Specialized agent for logistic regression tasks"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("LogisticRegression", model_path)
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Train the logistic regression model"""
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        self.model.fit(X, y)
        self.is_trained = True
        
        # Get feature importance
        importance = np.abs(self.model.coef_[0])
        feature_importance = dict(zip(X.columns, importance))
        
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
        y_prob = self.model.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1': f1_score(y, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y, y_prob)
        }

class RandomForestAgent(MLAgent):
    """Specialized agent for random forest tasks"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("RandomForest", model_path)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Train the random forest model"""
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
        y_prob = self.model.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1': f1_score(y, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y, y_prob)
        }

class XGBoostAgent(MLAgent):
    """Specialized agent for XGBoost tasks"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("XGBoost", model_path)
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    
    def train(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Train the XGBoost model"""
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
        y_prob = self.model.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='binary'),
            'recall': recall_score(y, y_pred, average='binary'),
            'f1': f1_score(y, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y, y_prob)
        } 