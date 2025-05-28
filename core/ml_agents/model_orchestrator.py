import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from pathlib import Path

class ModelOrchestrator:
    """Orchestrates ML model selection and task assignment"""
    
    def __init__(self, models_dir: str = "../models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Task type detection thresholds
        self.classification_threshold = 10  # Max unique values for classification
        self.clustering_min_rows = 1000  # Min rows for clustering
        
        # Initialize model registry
        self.model_registry = {
            'binary_classification': ['logistic_regression', 'random_forest', 'xgboost', 'decision_tree', 'knn', 'lightgbm'],
            'multiclass_classification': ['random_forest', 'xgboost', 'decision_tree', 'knn', 'lightgbm'],
            'regression': ['linear_regression', 'random_forest', 'xgboost', 'decision_tree', 'lightgbm'],
            'clustering': ['kmeans'],
            'recommendation': ['knn']
        }
        
        # Task-specific metrics
        self.task_metrics = {
            'binary_classification': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            'multiclass_classification': ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
            'regression': ['r2', 'mse', 'mae', 'rmse'],
            'clustering': ['silhouette', 'calinski_harabasz', 'davies_bouldin'],
            'recommendation': ['precision_at_k', 'recall_at_k', 'ndcg']
        }
    
    def detect_task_type(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Detect the type of ML task based on data characteristics"""
        task_info = {
            'task_type': None,
            'suitable_models': [],
            'target_type': None,
            'metrics': [],
            'preprocessing_steps': []
        }
        
        # Get target variable characteristics
        n_unique = data[target_col].nunique()
        is_numeric = pd.api.types.is_numeric_dtype(data[target_col])
        
        # Detect task type
        if is_numeric:
            if n_unique <= 2:  # Binary classification
                task_info.update({
                    'task_type': 'binary_classification',
                    'suitable_models': self.model_registry['binary_classification'],
                    'metrics': self.task_metrics['binary_classification'],
                    'preprocessing_steps': ['encoding', 'scaling', 'balancing']
                })
            elif n_unique <= self.classification_threshold:  # Multiclass classification
                task_info.update({
                    'task_type': 'multiclass_classification',
                    'suitable_models': self.model_registry['multiclass_classification'],
                    'metrics': self.task_metrics['multiclass_classification'],
                    'preprocessing_steps': ['encoding', 'scaling', 'balancing']
                })
            else:  # Regression
                task_info.update({
                    'task_type': 'regression',
                    'suitable_models': self.model_registry['regression'],
                    'metrics': self.task_metrics['regression'],
                    'preprocessing_steps': ['scaling', 'outlier_removal']
                })
        else:  # Categorical target
            if n_unique <= 2:  # Binary classification
                task_info.update({
                    'task_type': 'binary_classification',
                    'suitable_models': self.model_registry['binary_classification'],
                    'metrics': self.task_metrics['binary_classification'],
                    'preprocessing_steps': ['encoding', 'scaling', 'balancing']
                })
            else:  # Multiclass classification
                task_info.update({
                    'task_type': 'multiclass_classification',
                    'suitable_models': self.model_registry['multiclass_classification'],
                    'metrics': self.task_metrics['multiclass_classification'],
                    'preprocessing_steps': ['encoding', 'scaling', 'balancing']
                })
        
        # Check for clustering potential
        if len(data) >= self.clustering_min_rows:
            task_info['additional_tasks'] = [{
                'task_type': 'clustering',
                'suitable_models': self.model_registry['clustering'],
                'metrics': self.task_metrics['clustering'],
                'preprocessing_steps': ['scaling', 'dimensionality_reduction']
            }]
        
        # Check for recommendation potential
        if any(col.lower().contains('user') for col in data.columns) and \
           any(col.lower().contains('item') for col in data.columns):
            task_info['additional_tasks'] = task_info.get('additional_tasks', []) + [{
                'task_type': 'recommendation',
                'suitable_models': self.model_registry['recommendation'],
                'metrics': self.task_metrics['recommendation'],
                'preprocessing_steps': ['encoding', 'normalization']
            }]
        
        return task_info
    
    def prepare_data(self, data: pd.DataFrame, target_col: str, task_info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data based on detected task type"""
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Apply preprocessing steps
        for step in task_info['preprocessing_steps']:
            if step == 'encoding':
                # Encode categorical variables
                categorical_cols = X.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                
                if task_info['task_type'] in ['binary_classification', 'multiclass_classification']:
                    if not pd.api.types.is_numeric_dtype(y):
                        y = LabelEncoder().fit_transform(y.astype(str))
            
            elif step == 'scaling':
                # Scale numeric features
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
            
            elif step == 'balancing':
                if task_info['task_type'] in ['binary_classification', 'multiclass_classification']:
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(random_state=42)
                    X, y = smote.fit_resample(X, y)
            
            elif step == 'outlier_removal':
                if task_info['task_type'] == 'regression':
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(X.select_dtypes(include=[np.number])))
                    X = X[(z_scores < 3).all(axis=1)]
                    y = y[(z_scores < 3).all(axis=1)]
        
        return X, y
    
    def get_best_model(self, task_type: str, data_size: int) -> str:
        """Select the best model based on task type and data characteristics"""
        if task_type == 'binary_classification':
            if data_size < 10000:
                return 'logistic_regression'
            elif data_size < 100000:
                return 'random_forest'
            else:
                return 'lightgbm'
        
        elif task_type == 'multiclass_classification':
            if data_size < 10000:
                return 'random_forest'
            elif data_size < 100000:
                return 'xgboost'
            else:
                return 'lightgbm'
        
        elif task_type == 'regression':
            if data_size < 10000:
                return 'linear_regression'
            elif data_size < 100000:
                return 'random_forest'
            else:
                return 'lightgbm'
        
        elif task_type == 'clustering':
            return 'kmeans'
        
        elif task_type == 'recommendation':
            return 'knn'
        
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def translate_to_business_insight(self, model_results: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
        """Translate model results into business insights"""
        insights = []
        
        if task_type == 'binary_classification':
            insights.extend([
                {
                    'type': 'performance',
                    'description': f"Model achieves {model_results['accuracy']*100:.1f}% accuracy",
                    'impact': 'Decision-making reliability'
                },
                {
                    'type': 'precision',
                    'description': f"Precision of {model_results['precision']*100:.1f}% indicates low false positive rate",
                    'impact': 'Cost efficiency in interventions'
                },
                {
                    'type': 'recall',
                    'description': f"Recall of {model_results['recall']*100:.1f}% shows good detection rate",
                    'impact': 'Risk management effectiveness'
                }
            ])
            
            if 'feature_importance' in model_results:
                top_features = sorted(model_results['feature_importance'].items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
                insights.append({
                    'type': 'drivers',
                    'description': f"Top driving factors: {', '.join(f[0] for f in top_features)}",
                    'impact': 'Strategic focus areas'
                })
        
        elif task_type == 'regression':
            insights.extend([
                {
                    'type': 'accuracy',
                    'description': f"Model explains {model_results['r2']*100:.1f}% of variance",
                    'impact': 'Prediction reliability'
                },
                {
                    'type': 'error',
                    'description': f"Average prediction error: {model_results['mae']:.2f}",
                    'impact': 'Planning accuracy'
                }
            ])
            
            if 'feature_importance' in model_results:
                insights.append({
                    'type': 'drivers',
                    'description': "Key influencing factors identified",
                    'impact': 'Strategic planning'
                })
        
        elif task_type == 'clustering':
            insights.extend([
                {
                    'type': 'segments',
                    'description': f"Identified {model_results['n_clusters']} distinct segments",
                    'impact': 'Targeted strategies'
                },
                {
                    'type': 'quality',
                    'description': f"Clustering quality score: {model_results['silhouette']:.2f}",
                    'impact': 'Segmentation reliability'
                }
            ])
        
        return insights 