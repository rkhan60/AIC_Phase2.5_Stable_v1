from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from .base_agent import MLAgent

class KNNAgent(MLAgent):
    """Specialized agent for KNN-based recommendation tasks"""
    
    def __init__(self, model_path: Optional[str] = None, n_neighbors: int = 5):
        super().__init__("KNN", model_path)
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors,
            algorithm='auto',
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the KNN model"""
        # Scale the features
        scaled_data = self.scaler.fit_transform(data)
        
        # Train the model
        self.model.fit(scaled_data)
        self.is_trained = True
        self.feature_names = list(data.columns)
        
        # Save model
        self.save_model()
        
        return {
            'n_samples': len(data),
            'n_features': data.shape[1],
            'model_params': self.model.get_params()
        }
    
    def predict(self, data: pd.DataFrame, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for each data point"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Scale the query data
        scaled_data = self.scaler.transform(data)
        
        # Get distances and indices of nearest neighbors
        distances, indices = self.model.kneighbors(
            scaled_data,
            n_neighbors=k if k is not None else self.model.n_neighbors
        )
        
        return distances, indices
    
    def recommend(self, user_profile: pd.DataFrame, item_pool: pd.DataFrame, 
                 n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Generate recommendations for a user"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Find nearest items
        distances, indices = self.predict(user_profile, k=n_recommendations)
        
        # Get recommended items
        recommendations = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            item = item_pool.iloc[idx]
            recommendations.append({
                'item_id': item.name,
                'similarity_score': 1.0 / (1.0 + dist),
                'features': item.to_dict()
            })
        
        return sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)
    
    def analyze(self, data: pd.DataFrame, test_queries: pd.DataFrame) -> Dict[str, Any]:
        """Analyze recommendation performance"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Get recommendations for test queries
        distances, indices = self.predict(test_queries)
        
        # Calculate average distance to neighbors
        avg_distance = float(np.mean(distances))
        
        # Calculate coverage (unique items recommended)
        unique_recommendations = len(np.unique(indices))
        coverage = unique_recommendations / len(data)
        
        # Calculate diversity (average pairwise distance between recommendations)
        diversity = float(np.mean([
            np.mean([
                np.linalg.norm(data.iloc[i] - data.iloc[j])
                for j in indices[k] if i != j
            ])
            for k, i in enumerate(indices.flatten())
        ]))
        
        return {
            'avg_neighbor_distance': avg_distance,
            'coverage': coverage,
            'diversity': diversity,
            'unique_recommendations': unique_recommendations
        }
    
    def get_similar_items(self, item: pd.Series, n_similar: int = 5) -> List[Dict[str, Any]]:
        """Find similar items to a given item"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Reshape item to 2D array
        item_data = item.values.reshape(1, -1)
        
        # Scale the item data
        scaled_item = self.scaler.transform(item_data)
        
        # Find nearest neighbors
        distances, indices = self.model.kneighbors(scaled_item, n_neighbors=n_similar + 1)
        
        # Skip the first result (the item itself)
        similar_items = []
        for dist, idx in zip(distances[0][1:], indices[0][1:]):
            similar_items.append({
                'item_id': idx,
                'similarity_score': 1.0 / (1.0 + dist)
            })
        
        return similar_items
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on neighbor distances"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Calculate average impact of each feature on neighbor distances
        feature_impacts = []
        for i in range(len(self.feature_names)):
            # Create a modified version of the data with this feature perturbed
            perturbed_data = self.scaler.inverse_transform(
                self.model._fit_X.copy()
            )
            perturbed_data[:, i] = np.mean(perturbed_data[:, i])
            
            # Calculate new distances
            new_distances = self.model.kneighbors(
                self.scaler.transform(perturbed_data)
            )[0]
            
            # Impact is the change in average distance
            original_distances = self.model.kneighbors(self.model._fit_X)[0]
            impact = np.mean(np.abs(new_distances - original_distances))
            feature_impacts.append(impact)
        
        # Normalize impacts
        feature_impacts = np.array(feature_impacts)
        feature_impacts = feature_impacts / np.sum(feature_impacts)
        
        return dict(zip(self.feature_names, feature_impacts)) 