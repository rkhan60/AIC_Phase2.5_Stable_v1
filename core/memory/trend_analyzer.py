from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime, timedelta
import pandas as pd
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

@dataclass
class TrendInsight:
    """Structure for trend analysis insights"""
    metric_name: str
    trend_type: str  # 'increasing', 'decreasing', 'cyclic', 'anomaly'
    confidence: float
    magnitude: float
    period: Optional[float] = None  # For cyclic trends
    forecast: Optional[List[float]] = None
    related_metrics: List[str] = None
    model_performance: Dict[str, float] = None
    
class MemoryLSTM(nn.Module):
    """LSTM model for memory pattern prediction"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

class MemoryDataset(Dataset):
    """Dataset for memory metrics"""
    def __init__(self, data: np.ndarray, sequence_length: int):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length]
        return sequence, target

class AdvancedTrendAnalyzer:
    """Advanced trend analysis using ML techniques"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.pca = PCA(n_components=0.95)
        self.rf_model = RandomForestRegressor(n_estimators=100)
        self.nn_model = MLPRegressor(hidden_layer_sizes=(100, 50))
        self.lstm_model = None
        self.sequence_length = 10
        
    def analyze_trends(self, 
                      metrics_history: List[Dict[str, Any]],
                      target_metrics: Optional[List[str]] = None) -> Dict[str, TrendInsight]:
        """Perform comprehensive trend analysis"""
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if not target_metrics:
            target_metrics = [col for col in df.columns if col != 'timestamp']
            
        insights = {}
        for metric in target_metrics:
            if metric in df.columns:
                values = df[metric].values
                
                # Basic trend analysis
                trend_insight = self._analyze_single_metric(df, metric)
                
                # Train ML models
                if len(values) >= 30:  # Minimum data required for ML
                    lstm_performance = self.train_lstm_model(values)
                    ensemble_performance = self.train_ensemble_models(values)
                    
                    trend_insight.model_performance = {
                        'lstm': lstm_performance,
                        **ensemble_performance
                    }
                    
                insights[metric] = trend_insight
                
        # Find correlated metrics
        self._find_metric_relationships(df, insights)
        
        return insights
        
    def _analyze_single_metric(self, df: pd.DataFrame, metric: str) -> TrendInsight:
        """Analyze trends for a single metric"""
        values = df[metric].values
        timestamps = df['timestamp'].values
        
        # Detect trend type and magnitude
        trend_type, magnitude = self._detect_trend_type(values)
        
        # Check for cyclical patterns
        period = self._detect_periodicity(values) if trend_type != 'anomaly' else None
        
        # Generate forecast
        forecast = self._generate_forecast(values) if trend_type != 'anomaly' else None
        
        # Calculate confidence
        confidence = self._calculate_trend_confidence(values, trend_type)
        
        return TrendInsight(
            metric_name=metric,
            trend_type=trend_type,
            confidence=confidence,
            magnitude=magnitude,
            period=period,
            forecast=forecast
        )
        
    def _detect_trend_type(self, values: np.ndarray) -> Tuple[str, float]:
        """Detect the type and magnitude of trend"""
        # Remove NaN values
        values = values[~np.isnan(values)]
        
        # Check for anomalies
        anomaly_scores = self.anomaly_detector.fit_predict(values.reshape(-1, 1))
        if -1 in anomaly_scores:
            return 'anomaly', np.mean(np.abs(values))
            
        # Calculate trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.arange(len(values)),
            values
        )
        
        # Determine trend type and magnitude
        if p_value < 0.05:  # Statistically significant trend
            trend_type = 'increasing' if slope > 0 else 'decreasing'
            magnitude = abs(slope)
        else:
            trend_type = 'cyclic' if self._detect_periodicity(values) else 'stable'
            magnitude = std_err
            
        return trend_type, magnitude
        
    def _detect_periodicity(self, values: np.ndarray) -> Optional[float]:
        """Detect periodic patterns using FFT"""
        if len(values) < 4:  # Need minimum points for FFT
            return None
            
        # Compute FFT
        fft = np.fft.fft(values)
        freqs = np.fft.fftfreq(len(values))
        
        # Find dominant frequency
        main_freq_idx = np.argmax(np.abs(fft[1:])) + 1
        if np.abs(fft[main_freq_idx]) > 0.1 * len(values):
            return 1 / freqs[main_freq_idx]
            
        return None
        
    def _generate_forecast(self, values: np.ndarray, horizon: int = 10) -> List[float]:
        """Generate simple forecast using exponential smoothing"""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        # Fit model
        model = ExponentialSmoothing(
            values,
            seasonal_periods=12,
            trend='add',
            seasonal='add'
        ).fit()
        
        # Generate forecast
        forecast = model.forecast(horizon)
        return forecast.tolist()
        
    def _calculate_trend_confidence(self, values: np.ndarray, trend_type: str) -> float:
        """Calculate confidence in trend detection"""
        if trend_type == 'anomaly':
            # Use isolation forest score
            return abs(np.mean(self.anomaly_detector.score_samples(values.reshape(-1, 1))))
            
        # Calculate confidence based on trend stability
        if len(values) < 2:
            return 0.5
            
        # Calculate rolling statistics
        rolling_mean = pd.Series(values).rolling(window=3).mean()
        rolling_std = pd.Series(values).rolling(window=3).std()
        
        # Confidence based on stability
        stability = 1 - (rolling_std.mean() / (rolling_mean.mean() + 1e-10))
        return max(0.0, min(1.0, stability))
        
    def _find_metric_relationships(self, 
                                 df: pd.DataFrame,
                                 insights: Dict[str, TrendInsight]):
        """Find relationships between metrics using PCA"""
        # Prepare data
        metric_cols = [col for col in df.columns if col != 'timestamp']
        if len(metric_cols) < 2:
            return
            
        X = df[metric_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        pca_result = self.pca.fit_transform(X_scaled)
        
        # Find related metrics based on component weights
        for i, metric in enumerate(metric_cols):
            related = []
            for j, weights in enumerate(self.pca.components_):
                if abs(weights[i]) > 0.3:  # Significant correlation threshold
                    # Find other metrics with similar weights in this component
                    related.extend([
                        metric_cols[k] for k, w in enumerate(weights)
                        if k != i and abs(w) > 0.3
                    ])
            
            if metric in insights:
                insights[metric].related_metrics = list(set(related))
                
    def detect_anomalies(self, 
                        metrics_history: List[Dict[str, Any]],
                        sensitivity: float = 0.1) -> Dict[str, List[datetime]]:
        """Detect anomalies in metrics"""
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        anomalies = {}
        for column in df.columns:
            if column != 'timestamp':
                # Use DBSCAN for anomaly detection
                X = df[column].values.reshape(-1, 1)
                X_scaled = self.scaler.fit_transform(X)
                
                clustering = DBSCAN(eps=sensitivity, min_samples=3).fit(X_scaled)
                
                # Points labeled as -1 are anomalies
                anomaly_indices = np.where(clustering.labels_ == -1)[0]
                if len(anomaly_indices) > 0:
                    anomalies[column] = df['timestamp'].iloc[anomaly_indices].tolist()
                    
        return anomalies
        
    def analyze_seasonal_patterns(self, 
                                metrics_history: List[Dict[str, Any]],
                                metric: str) -> Dict[str, Any]:
        """Analyze seasonal patterns in metric"""
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if metric not in df.columns:
            return {}
            
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        patterns = {
            'hourly_pattern': df.groupby('hour')[metric].mean().to_dict(),
            'daily_pattern': df.groupby('day')[metric].mean().to_dict(),
            'weekly_pattern': df.groupby('day_of_week')[metric].mean().to_dict()
        }
        
        # Detect significant patterns
        significant_patterns = {}
        for period, values in patterns.items():
            # Calculate variation
            values_array = np.array(list(values.values()))
            if np.std(values_array) > 0.1 * np.mean(values_array):
                significant_patterns[period] = values
                
        return significant_patterns
        
    def analyze_trend_changes(self,
                            metrics_history: List[Dict[str, Any]],
                            metric: str,
                            window_size: int = 10) -> List[Dict[str, Any]]:
        """Detect significant changes in trends"""
        df = pd.DataFrame(metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if metric not in df.columns:
            return []
            
        changes = []
        values = df[metric].values
        
        # Use rolling windows to detect trend changes
        for i in range(window_size, len(values) - window_size):
            window1 = values[i - window_size:i]
            window2 = values[i:i + window_size]
            
            # Calculate trends in both windows
            slope1 = np.polyfit(np.arange(window_size), window1, 1)[0]
            slope2 = np.polyfit(np.arange(window_size), window2, 1)[0]
            
            # Check for significant change
            if abs(slope1 - slope2) > np.std(values) * 0.5:
                changes.append({
                    'timestamp': df['timestamp'].iloc[i],
                    'previous_trend': slope1,
                    'new_trend': slope2,
                    'magnitude': abs(slope1 - slope2)
                })
                
        return changes
        
    def train_lstm_model(self, values: np.ndarray) -> Dict[str, float]:
        """Train LSTM model for sequence prediction"""
        # Prepare data
        values = values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)
        
        # Create dataset
        dataset = MemoryDataset(scaled_values, self.sequence_length)
        train_size = int(0.8 * len(dataset))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Initialize model
        self.lstm_model = MemoryLSTM(1, 64, 2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.lstm_model.parameters())
        
        # Training
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(50):
            self.lstm_model.train()
            train_loss = 0
            for sequences, targets in train_loader:
                optimizer.zero_grad()
                outputs = self.lstm_model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # Validation
            self.lstm_model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, targets in val_loader:
                    outputs = self.lstm_model(sequences)
                    val_loss += criterion(outputs, targets).item()
                    
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
        return {
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader)
        }
        
    def train_ensemble_models(self, values: np.ndarray) -> Dict[str, float]:
        """Train multiple models for ensemble prediction"""
        # Prepare data
        X = np.arange(len(values)).reshape(-1, 1)
        y = values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Train Random Forest
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        rf_score = r2_score(y_test, rf_pred)
        
        # Train Neural Network
        self.nn_model.fit(X_train, y_train)
        nn_pred = self.nn_model.predict(X_test)
        nn_score = r2_score(y_test, nn_pred)
        
        return {
            'random_forest_r2': rf_score,
            'neural_network_r2': nn_score
        }
        
    def generate_ensemble_forecast(self, 
                                 values: np.ndarray, 
                                 horizon: int = 10) -> Dict[str, List[float]]:
        """Generate forecasts using multiple models"""
        X = np.arange(len(values)).reshape(-1, 1)
        future_X = np.arange(len(values), len(values) + horizon).reshape(-1, 1)
        
        forecasts = {
            'rf_forecast': self.rf_model.predict(future_X).tolist(),
            'nn_forecast': self.nn_model.predict(future_X).tolist()
        }
        
        if self.lstm_model is not None:
            # Prepare sequence for LSTM
            sequence = torch.FloatTensor(
                self.scaler.transform(values[-self.sequence_length:].reshape(-1, 1))
            ).unsqueeze(0)
            
            self.lstm_model.eval()
            with torch.no_grad():
                lstm_forecast = []
                for _ in range(horizon):
                    pred = self.lstm_model(sequence).item()
                    lstm_forecast.append(pred)
                    sequence = torch.cat([
                        sequence[:, 1:], 
                        torch.FloatTensor([[pred]])
                    ], dim=1)
                    
            forecasts['lstm_forecast'] = self.scaler.inverse_transform(
                np.array(lstm_forecast).reshape(-1, 1)
            ).flatten().tolist()
            
        return forecasts 