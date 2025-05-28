import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from .base_agent import MLAgent

class DataQualityAgent(MLAgent):
    """Agent specialized in data quality assessment and cleaning using ML"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__("Data Quality", model_path)
        self.scaler = StandardScaler()
        self.quality_scores: Dict[str, float] = {}
        self.column_profiles: Dict[str, Dict] = {}
        
    def train(self, data: pd.DataFrame) -> None:
        """Train the anomaly detection model for quality assessment"""
        self.validate_data(data)
        numeric_data = data.select_dtypes(include=[np.number])
        
        if not numeric_data.empty:
            # Scale the numeric data
            scaled_data = self.scaler.fit_transform(numeric_data)
            
            # Train Isolation Forest for anomaly detection
            self.model = IsolationForest(contamination=0.1, random_state=42)
            self.model.fit(scaled_data)
            self.is_trained = True
            
            # Save the model
            self.save_model()
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Predict data quality anomalies"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return np.zeros(len(data))
        
        scaled_data = self.scaler.transform(numeric_data)
        return self.model.predict(scaled_data)
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data quality analysis"""
        self.validate_data(data)
        analysis = {
            'quality_scores': {},
            'column_profiles': {},
            'anomalies': {},
            'recommendations': []
        }
        
        # Profile each column
        for col in data.columns:
            profile = self._analyze_column(data[col])
            analysis['column_profiles'][col] = profile
            analysis['quality_scores'][col] = self._calculate_quality_score(profile)
        
        # Detect anomalies if model is trained
        if self.is_trained:
            anomalies = self.predict(data)
            analysis['anomalies'] = {
                'count': int((anomalies == -1).sum()),
                'percentage': float((anomalies == -1).mean() * 100),
                'indices': list(np.where(anomalies == -1)[0])
            }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze a single column"""
        profile = {
            'type': str(series.dtype),
            'unique_count': int(series.nunique()),
            'missing_count': int(series.isnull().sum()),
            'missing_percentage': float(series.isnull().mean() * 100)
        }
        
        if np.issubdtype(series.dtype, np.number):
            profile.update({
                'mean': float(series.mean()) if not series.isnull().all() else None,
                'std': float(series.std()) if not series.isnull().all() else None,
                'min': float(series.min()) if not series.isnull().all() else None,
                'max': float(series.max()) if not series.isnull().all() else None,
                'zeros_percentage': float((series == 0).mean() * 100),
                'negative_percentage': float((series < 0).mean() * 100)
            })
        elif series.dtype == 'object':
            profile.update({
                'empty_string_count': int((series == '').sum()),
                'avg_length': float(series.str.len().mean()) if not series.isnull().all() else 0,
                'common_values': series.value_counts().head(5).to_dict()
            })
        
        return profile
    
    def _calculate_quality_score(self, profile: Dict[str, Any]) -> float:
        """Calculate quality score for a column"""
        score = 100.0
        deductions = []
        
        # Deduct for missing values
        missing_deduction = profile.get('missing_percentage', 0) * 0.5
        deductions.append(missing_deduction)
        
        # Deduct for low uniqueness if applicable
        if profile.get('unique_count', 0) > 0:
            uniqueness_ratio = profile.get('unique_count', 0) / (profile.get('unique_count', 0) + profile.get('missing_count', 0))
            if uniqueness_ratio < 0.1:
                deductions.append(20.0)
        
        # Type-specific deductions
        if profile.get('type') == 'object':
            if profile.get('empty_string_count', 0) > 0:
                deductions.append(profile.get('empty_string_count', 0) * 0.1)
        elif 'float' in str(profile.get('type', '')):
            if profile.get('zeros_percentage', 0) > 50:
                deductions.append(10.0)
            if profile.get('negative_percentage', 0) > 10:
                deductions.append(5.0)
        
        # Calculate final score
        final_score = score - sum(deductions)
        return max(0.0, min(100.0, final_score))
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Check overall quality
        low_quality_cols = [
            col for col, score in analysis['quality_scores'].items()
            if score < 70
        ]
        
        if low_quality_cols:
            recommendations.append({
                'type': 'quality_improvement',
                'priority': 'high',
                'description': f"Improve data quality for columns: {', '.join(low_quality_cols)}",
                'impact': 'Data reliability and analysis accuracy'
            })
        
        # Check for missing values
        high_missing_cols = [
            col for col, profile in analysis['column_profiles'].items()
            if profile.get('missing_percentage', 0) > 20
        ]
        
        if high_missing_cols:
            recommendations.append({
                'type': 'missing_values',
                'priority': 'high',
                'description': f"Address high missing values in: {', '.join(high_missing_cols)}",
                'impact': 'Data completeness and reliability'
            })
        
        # Check for anomalies
        if analysis.get('anomalies', {}).get('percentage', 0) > 5:
            recommendations.append({
                'type': 'anomalies',
                'priority': 'medium',
                'description': f"Investigate {analysis['anomalies']['count']} anomalous records",
                'impact': 'Data quality and reliability'
            })
        
        return recommendations
    
    def clean_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean the data based on quality analysis"""
        analysis = self.analyze(data)
        cleaned_data = data.copy()
        cleaning_report = {
            'rows_before': len(data),
            'columns_before': len(data.columns),
            'actions_taken': []
        }
        
        # Remove or fix anomalies
        if self.is_trained:
            anomalies = self.predict(data)
            cleaned_data = cleaned_data[anomalies != -1]
            if len(cleaned_data) < len(data):
                cleaning_report['actions_taken'].append({
                    'action': 'remove_anomalies',
                    'details': f"Removed {len(data) - len(cleaned_data)} anomalous rows"
                })
        
        # Handle missing values
        for col, profile in analysis['column_profiles'].items():
            if profile['missing_percentage'] > 0:
                if np.issubdtype(cleaned_data[col].dtype, np.number):
                    # Fill numeric missing values with median
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
                    cleaning_report['actions_taken'].append({
                        'action': 'fill_missing',
                        'column': col,
                        'method': 'median'
                    })
                else:
                    # Fill categorical missing values with mode
                    cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mode()[0])
                    cleaning_report['actions_taken'].append({
                        'action': 'fill_missing',
                        'column': col,
                        'method': 'mode'
                    })
        
        # Update cleaning report
        cleaning_report.update({
            'rows_after': len(cleaned_data),
            'columns_after': len(cleaned_data.columns),
            'rows_removed': len(data) - len(cleaned_data),
            'quality_improvement': {
                col: analysis['quality_scores'][col]
                for col in cleaned_data.columns
            }
        })
        
        return cleaned_data, cleaning_report 