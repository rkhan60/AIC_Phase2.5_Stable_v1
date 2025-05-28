import pandas as pd
import dask.dataframe as dd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import tableauserverclient as TSC
from powerbiclient import QuickVisualize, Report
from azure.identity import InteractiveBrowserCredential
import os
from dotenv import load_dotenv
import warnings
import multiprocessing as mp
from functools import lru_cache
import gc

class CompanyDataAnalyzer:
    def __init__(self, processing_mode: str = "Memory Efficient"):
        self.processing_mode = processing_mode
        self.encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        self.chunk_size = 100000  # Process 100k rows at a time
        self.summary_stats = {}
        self.insights = {}
        self.cache = {}
        self.n_cores = mp.cpu_count()
        
        # Smart data type inference
        self.type_patterns = {
            'id': 'object',  # Fields containing 'id' are usually strings
            'name': 'object',  # Name fields are strings
            'code': 'object',  # Code fields are strings
            'count': 'float64',  # Count fields are numeric
            'amount': 'float64',  # Amount fields are numeric
            'quantity': 'float64',  # Quantity fields are numeric
            'date': 'datetime64[ns]',  # Date fields
            'timestamp': 'datetime64[ns]'  # Timestamp fields
        }
        
        # Data types for ECRM
        self.ecrm_dtypes = {
            'AccountNumber': 'float64',
            'Lead.Id': 'object',
            'Lead.Name': 'object',
            'cdi_bouncescount': 'float64',
            'cdi_clickscount': 'float64',
            'cdi_deliveriescount': 'float64',
            'cdi_messagecount': 'float64',
            'cdi_openscount': 'float64',
            'cdi_uniqueclickscount': 'float64',
            'cdi_uniqueopenscount': 'float64',
            'cdi_unsubscribescount': 'float64'
        }
        
        # Data types for SCRM
        self.scrm_dtypes = {
            'Quantity.Ordered': 'float64',
            'familycode': 'float64',
            'wc_zinaicscode': 'object',
            'zisiccode': 'object'
        }
        
        load_dotenv()  # Load environment variables for API credentials
        warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output
        
    def _infer_column_types(self, columns):
        """Intelligently infer column types based on patterns."""
        dtypes = {}
        for col in columns:
            col_lower = col.lower()
            # Match column name against patterns
            matched = False
            for pattern, dtype in self.type_patterns.items():
                if pattern in col_lower:
                    dtypes[col] = dtype
                    matched = True
                    break
            # Default to float64 for unmatched numeric-looking columns
            if not matched:
                dtypes[col] = 'float64'
        return dtypes
    
    @lru_cache(maxsize=32)
    def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get cached file information."""
        path = Path(file_path)
        return {
            'size': path.stat().st_size / (1024 * 1024 * 1024),
            'modified': path.stat().st_mtime,
            'name': path.name
        }
    
    def _parallel_process(self, df: pd.DataFrame, func, *args):
        """Process data in parallel using all available cores."""
        if len(df) < 1000:  # Small dataset, process normally
            return func(df, *args)
        
        # Split dataframe into chunks
        chunks = np.array_split(df, self.n_cores)
        
        # Process chunks in parallel
        with mp.Pool(self.n_cores) as pool:
            results = pool.starmap(func, [(chunk, *args) for chunk in chunks])
        
        # Combine results
        return pd.concat(results)
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect the correct encoding of a file."""
        for encoding in self.encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Try reading a small chunk
                return encoding
            except UnicodeDecodeError:
                continue
        return 'utf-8'  # Default to UTF-8 if no encoding works
    
    def _process_chunks(self, chunks) -> pd.DataFrame:
        """Process data chunks and combine them."""
        print("📦 Processing data chunks...")
        dfs = []
        total_rows = 0
        
        for chunk in chunks:
            dfs.append(chunk)
            total_rows += len(chunk)
            print(f"✓ Processed {total_rows:,} rows", end='\r')
        
        print("\n✨ Combining chunks...")
        return pd.concat(dfs, ignore_index=True)

    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """Load data with smart type inference and parallel processing."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get cached file info
        file_info = self._get_file_info(str(file_path))
        print(f"\n📊 Loading file: {file_info['name']}")
        print(f"📦 File size: {file_info['size']:.2f} GB")
        
        # Detect encoding
        encoding = self._detect_encoding(file_path)
        print(f"📝 Using encoding: {encoding}")
        
        try:
            # Read a small sample to infer types
            sample = pd.read_csv(file_path, nrows=1000, encoding=encoding)
            dtypes = self._infer_column_types(sample.columns)
            
            # Add float type for known numeric columns
            numeric_columns = [
                'Quantity.Ordered', 'Line.Item.Amount', 'Revenue',
                'cdi_bouncescount', 'cdi_clickscount', 'cdi_deliveriescount',
                'cdi_messagecount', 'cdi_openscount', 'cdi_uniqueclickscount',
                'cdi_uniqueopenscount', 'cdi_unsubscribescount',
                'Semantic_Confidence_Score'
            ]
            
            for col in numeric_columns:
                if col in dtypes:
                    dtypes[col] = 'float64'
            
            if file_info['size'] > 1.0:
                print("🔄 Using chunked reading for large file...")
                chunks = pd.read_csv(file_path, dtype=dtypes, chunksize=self.chunk_size,
                                   encoding=encoding, na_values=['', 'NA', 'null', 'NULL', 'NaN'])
                return self._process_chunks(chunks)
            else:
                print("📝 Reading with pandas...")
                return pd.read_csv(file_path, dtype=dtypes, encoding=encoding,
                                 na_values=['', 'NA', 'null', 'NULL', 'NaN'])
                
        except Exception as e:
            print(f"⚠️ Error reading file: {str(e)}")
            print("🔄 Attempting to read without type inference...")
            try:
                if file_info['size'] > 1.0:
                    chunks = pd.read_csv(file_path, chunksize=self.chunk_size,
                                       encoding=encoding, na_values=['', 'NA', 'null', 'NULL', 'NaN'])
                    df = self._process_chunks(chunks)
                else:
                    df = pd.read_csv(file_path, encoding=encoding,
                                   na_values=['', 'NA', 'null', 'NULL', 'NaN'])
                
                # Convert numeric columns after loading
                for col in df.columns:
                    if col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            except Exception as e2:
                raise Exception(f"Failed to read file: {str(e2)}")
    
    def _calculate_stats(self, data):
        """Calculate basic statistics for a numeric series."""
        try:
            # Convert to numeric if needed
            numeric_data = pd.to_numeric(data, errors='coerce')
            numeric_data = numeric_data.dropna()
            
            if len(numeric_data) == 0:
                return pd.Series({
                    'mean': None,
                    'median': None,
                    'std': None,
                    'min': None,
                    'max': None
                })
            
            return pd.Series({
                'mean': float(numeric_data.mean()),
                'median': float(numeric_data.median()),
                'std': float(numeric_data.std()),
                'min': float(numeric_data.min()),
                'max': float(numeric_data.max())
            })
        except Exception as e:
            print(f"⚠️ Error calculating stats: {str(e)}")
            return pd.Series({
                'mean': None,
                'median': None,
                'std': None,
                'min': None,
                'max': None,
                'error': str(e)
            })
    
    def _detect_outliers(self, data):
        """Detect outliers in a numeric series."""
        try:
            # Convert to numeric, coercing errors to NaN
            numeric_data = pd.to_numeric(data, errors='coerce')
            # Drop NaN values
            numeric_data = numeric_data.dropna()
            
            if len(numeric_data) == 0:
                return pd.Series({
                    'outlier_count': 0,
                    'outlier_percentage': 0,
                    'min_outlier': None,
                    'max_outlier': None
                })
            
            mean = numeric_data.mean()
            std = numeric_data.std()
            mask = abs(numeric_data - mean) > 3 * std
            outliers = numeric_data[mask]
            
            return pd.Series({
                'outlier_count': int(len(outliers)),
                'outlier_percentage': float((len(outliers) / len(numeric_data)) * 100),
                'min_outlier': float(outliers.min()) if len(outliers) > 0 else None,
                'max_outlier': float(outliers.max()) if len(outliers) > 0 else None
            })
        except Exception as e:
            print(f"⚠️ Error in outlier detection: {str(e)}")
            return pd.Series({
                'outlier_count': 0,
                'outlier_percentage': 0,
                'min_outlier': None,
                'max_outlier': None,
                'error': str(e)
            })
    
    def _convert_to_serializable(self, obj):
        """Convert numpy and pandas types to Python native types."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, pd.core.dtypes.base.ExtensionDtype):
            return str(obj)
        elif isinstance(obj, type):
            return str(obj)
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)  # Convert both Python and numpy boolean types
        else:
            try:
                # Try standard JSON serialization
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                # If standard serialization fails, convert to string
                return str(obj)
    
    def _profile_data(self, df: pd.DataFrame) -> Dict:
        """Profile the data to identify patterns and potential issues."""
        profile = {
            'column_profiles': {},
            'data_quality_score': {},
            'validation_results': {},
            'recommendations': []
        }
        
        print("📊 Profiling data...")
        
        # Profile each column
        for col in df.columns:
            try:
                col_profile = {
                    'type': str(df[col].dtype),
                    'unique_count': int(df[col].nunique()),
                    'missing_count': int(df[col].isnull().sum()),
                    'missing_percentage': float(df[col].isnull().mean() * 100)
                }
                
                # Identify data type specific patterns
                if df[col].dtype == 'object':
                    # String patterns
                    sample = df[col].dropna().head(1000)
                    col_profile.update({
                        'avg_length': float(sample.str.len().mean()),
                        'contains_numbers': bool(sample.str.contains(r'\d', regex=True).any()),
                        'contains_special_chars': bool(sample.str.contains(r'[^a-zA-Z0-9\s]', regex=True).any()),
                        'common_patterns': self._identify_patterns(sample)
                    })
                elif np.issubdtype(df[col].dtype, np.number):
                    # Numeric patterns
                    numeric_data = pd.to_numeric(df[col], errors='coerce')
                    col_profile.update({
                        'zero_count': int((numeric_data == 0).sum()),
                        'negative_count': int((numeric_data < 0).sum()),
                        'distribution_type': self._detect_distribution(numeric_data)
                    })
                
                profile['column_profiles'][col] = col_profile
                
                # Generate quality score
                profile['data_quality_score'][col] = self._calculate_quality_score(col_profile)
                
                # Validate data
                profile['validation_results'][col] = self._validate_column(df[col], col_profile)
                
            except Exception as e:
                print(f"⚠️ Error profiling column {col}: {str(e)}")
                profile['column_profiles'][col] = {'error': str(e)}
        
        # Generate recommendations
        profile['recommendations'] = self._generate_recommendations(profile)
        
        return profile
    
    def _identify_patterns(self, series: pd.Series) -> Dict:
        """Identify common patterns in string data."""
        patterns = {
            'email': r'^[^@]+@[^@]+\.[^@]+$',
            'url': r'^https?://',
            'date': r'^\d{2,4}[-/]\d{2}[-/]\d{2,4}$',
            'phone': r'^\+?[\d\-\(\)\s]{10,}$',
            'alphanumeric': r'^[a-zA-Z0-9]+$',
            'numeric_only': r'^\d+$',
            'alpha_only': r'^[a-zA-Z]+$'
        }
        
        results = {}
        for name, pattern in patterns.items():
            try:
                match_count = series.str.match(pattern, na=False).sum()
                if match_count > 0:
                    results[name] = int(match_count)
            except Exception as e:
                print(f"⚠️ Error matching pattern {name}: {str(e)}")
        
        return results
    
    def _detect_distribution(self, series: pd.Series) -> str:
        """Detect the likely distribution type of numeric data."""
        try:
            # Remove NaN values
            clean_data = series.dropna()
            if len(clean_data) < 10:
                return 'insufficient_data'
            
            # Calculate basic statistics
            skewness = float(clean_data.skew())
            kurtosis = float(clean_data.kurtosis())
            
            # Simple distribution detection based on skewness and kurtosis
            if abs(skewness) < 0.5 and abs(kurtosis) < 3:
                return 'normal'
            elif skewness > 1:
                return 'right_skewed'
            elif skewness < -1:
                return 'left_skewed'
            elif kurtosis > 3:
                return 'heavy_tailed'
            else:
                return 'unknown'
        except:
            return 'error'
    
    def _calculate_quality_score(self, profile: Dict) -> float:
        """Calculate a quality score for a column based on its profile."""
        score = 100.0
        deductions = {
            'missing_data': profile.get('missing_percentage', 0) * 0.5,
            'low_uniqueness': 0.0,
            'pattern_mismatch': 0.0,
            'data_errors': 0.0
        }
        
        # Deduct for low uniqueness if applicable
        if profile.get('unique_count', 0) > 0:
            expected_unique = profile.get('unique_count', 0) / 2
            if profile.get('unique_count', 0) < expected_unique:
                deductions['low_uniqueness'] = 20.0
        
        # Deduct for pattern mismatches
        if 'common_patterns' in profile:
            pattern_matches = sum(profile['common_patterns'].values())
            total_rows = profile.get('unique_count', 0)
            if total_rows > 0 and pattern_matches / total_rows < 0.8:
                deductions['pattern_mismatch'] = 15.0
        
        # Deduct for data errors
        if profile.get('error'):
            deductions['data_errors'] = 50.0
        
        # Calculate final score
        final_score = score - sum(deductions.values())
        return max(0.0, min(100.0, final_score))
    
    def _validate_column(self, series: pd.Series, profile: Dict) -> Dict:
        """Validate column data based on its profile."""
        validation = {
            'issues': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Check for basic issues
        if profile.get('missing_percentage', 0) > 20:
            validation['issues'].append(f"High missing data: {profile['missing_percentage']:.1f}%")
        
        if profile.get('unique_count', 0) == 1:
            validation['warnings'].append("Column has only one unique value")
        
        # Type-specific validation
        if profile.get('type') == 'object':
            if profile.get('contains_numbers') and profile.get('contains_special_chars'):
                validation['warnings'].append("Mixed content types detected")
            
            patterns = profile.get('common_patterns', {})
            if patterns.get('email', 0) > 0 and patterns.get('email') < len(series):
                validation['warnings'].append("Inconsistent email format")
            if patterns.get('date', 0) > 0 and patterns.get('date') < len(series):
                validation['warnings'].append("Inconsistent date format")
        
        elif 'float' in str(profile.get('type', '')):
            if profile.get('zero_count', 0) > len(series) * 0.5:
                validation['warnings'].append("High number of zero values")
            if profile.get('negative_count', 0) > 0:
                validation['warnings'].append("Contains negative values")
        
        # Add suggestions
        if profile.get('missing_percentage', 0) > 0:
            validation['suggestions'].append("Consider imputing missing values")
        if profile.get('unique_count', 0) < len(series) * 0.01:
            validation['suggestions'].append("Consider encoding as categorical")
        
        return validation
    
    def _generate_recommendations(self, profile: Dict) -> List[Dict]:
        """Generate recommendations based on the data profile."""
        recommendations = []
        
        # Analyze quality scores
        low_quality_cols = [
            col for col, score in profile['data_quality_score'].items()
            if score < 70
        ]
        if low_quality_cols:
            recommendations.append({
                'type': 'quality_improvement',
                'priority': 'high',
                'description': f"Improve data quality for columns: {', '.join(low_quality_cols)}",
                'impact': 'Data reliability and analysis accuracy'
            })
        
        # Analyze validation results
        for col, validation in profile['validation_results'].items():
            if validation['issues']:
                recommendations.append({
                    'type': 'data_cleaning',
                    'priority': 'high',
                    'description': f"Address data issues in {col}: {'; '.join(validation['issues'])}",
                    'impact': 'Data integrity'
                })
        
        # Analyze patterns
        for col, col_profile in profile['column_profiles'].items():
            if 'common_patterns' in col_profile:
                patterns = col_profile['common_patterns']
                if patterns.get('date', 0) > 0:
                    recommendations.append({
                        'type': 'type_conversion',
                        'priority': 'medium',
                        'description': f"Convert {col} to datetime type",
                        'impact': 'Better date handling and analysis'
                    })
        
        return recommendations
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze trends in the data."""
        trends = {
            'temporal_patterns': {},
            'value_patterns': {},
            'sequence_patterns': {},
            'cyclical_patterns': {}
        }
        
        print("📈 Analyzing trends...")
        
        # Analyze temporal patterns if date columns exist
        date_columns = []
        for col in df.columns:
            try:
                # Check if column contains dates
                if df[col].dtype == 'object':
                    sample = df[col].dropna().head(1000)
                    if self._identify_patterns(sample).get('date', 0) > 0:
                        date_columns.append(col)
                elif pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_columns.append(col)
            except:
                continue
        
        if date_columns:
            print("📅 Analyzing temporal patterns...")
            for col in date_columns:
                try:
                    # Convert to datetime if needed
                    dates = pd.to_datetime(df[col], errors='coerce')
                    if not dates.isna().all():
                        temporal_stats = {
                            'min_date': dates.min().isoformat(),
                            'max_date': dates.max().isoformat(),
                            'date_range_days': (dates.max() - dates.min()).days,
                            'missing_dates': int(dates.isna().sum()),
                            'unique_dates': int(dates.nunique())
                        }
                        
                        # Analyze daily patterns
                        daily_counts = dates.dt.day_name().value_counts()
                        temporal_stats['daily_pattern'] = daily_counts.to_dict()
                        
                        # Analyze monthly patterns
                        monthly_counts = dates.dt.month_name().value_counts()
                        temporal_stats['monthly_pattern'] = monthly_counts.to_dict()
                        
                        # Analyze yearly patterns
                        yearly_counts = dates.dt.year.value_counts()
                        temporal_stats['yearly_pattern'] = yearly_counts.to_dict()
                        
                        trends['temporal_patterns'][col] = temporal_stats
                except Exception as e:
                    print(f"⚠️ Error analyzing temporal patterns for {col}: {str(e)}")
        
        # Analyze value patterns in numeric columns
        print("📊 Analyzing value patterns...")
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            try:
                series = pd.to_numeric(df[col], errors='coerce')
                if not series.isna().all():
                    # Calculate basic trend indicators
                    diffs = series.diff()
                    value_stats = {
                        'trend_direction': 'increasing' if diffs.mean() > 0 else 'decreasing',
                        'trend_strength': abs(diffs.mean() / series.std()) if series.std() != 0 else 0,
                        'volatility': float(series.std() / series.mean()) if series.mean() != 0 else 0,
                        'has_seasonality': bool(self._check_seasonality(series))
                    }
                    
                    # Detect patterns
                    patterns = []
                    if value_stats['trend_strength'] > 0.1:
                        patterns.append(f"{value_stats['trend_direction']} trend")
                    if value_stats['volatility'] > 0.5:
                        patterns.append('high volatility')
                    if value_stats['has_seasonality']:
                        patterns.append('seasonal pattern')
                    
                    value_stats['detected_patterns'] = patterns
                    trends['value_patterns'][col] = value_stats
            except Exception as e:
                print(f"⚠️ Error analyzing value patterns for {col}: {str(e)}")
        
        # Analyze sequence patterns
        print("🔄 Analyzing sequence patterns...")
        for col in df.columns:
            try:
                sequence_stats = {
                    'repeating_values': {},
                    'value_transitions': {},
                    'sequence_length': len(df)
                }
                
                # Analyze repeating values
                value_counts = df[col].value_counts()
                if len(value_counts) > 0:
                    sequence_stats['repeating_values'] = {
                        str(k): int(v) for k, v in value_counts.head(5).items()
                    }
                
                # Analyze value transitions
                if len(df) > 1:
                    transitions = pd.DataFrame({
                        'from': df[col][:-1],
                        'to': df[col][1:]
                    }).value_counts().head(5)
                    
                    sequence_stats['value_transitions'] = {
                        f"{k[0]} -> {k[1]}": int(v) for k, v in transitions.items()
                    }
                
                trends['sequence_patterns'][col] = sequence_stats
            except Exception as e:
                print(f"⚠️ Error analyzing sequence patterns for {col}: {str(e)}")
        
        # Analyze cyclical patterns
        print("🔄 Analyzing cyclical patterns...")
        for col in numeric_columns:
            try:
                series = pd.to_numeric(df[col], errors='coerce')
                if not series.isna().all() and len(series) > 10:
                    # Detect cycles using autocorrelation
                    autocorr = series.autocorr()
                    
                    cyclical_stats = {
                        'has_cycles': abs(autocorr) > 0.7,
                        'cycle_strength': float(abs(autocorr)),
                        'cycle_length': self._estimate_cycle_length(series)
                    }
                    
                    trends['cyclical_patterns'][col] = cyclical_stats
            except Exception as e:
                print(f"⚠️ Error analyzing cyclical patterns for {col}: {str(e)}")
        
        return trends
    
    def _check_seasonality(self, series: pd.Series) -> bool:
        """Check if a series has seasonal patterns."""
        try:
            if len(series) < 4:
                return False
            
            # Calculate autocorrelation at different lags
            autocorr = [series.autocorr(lag) for lag in range(1, min(len(series)//2, 13))]
            
            # Check for repeating patterns in autocorrelation
            peaks = [i for i in range(1, len(autocorr)-1)
                    if autocorr[i-1] < autocorr[i] > autocorr[i+1]]
            
            return len(peaks) > 0 and max([autocorr[i] for i in peaks]) > 0.7
        except:
            return False
    
    def _estimate_cycle_length(self, series: pd.Series) -> Optional[int]:
        """Estimate the length of cycles in a series."""
        try:
            if len(series) < 4:
                return None
            
            # Calculate autocorrelation at different lags
            autocorr = [series.autocorr(lag) for lag in range(1, min(len(series)//2, 25))]
            
            # Find peaks in autocorrelation
            peaks = [i for i in range(1, len(autocorr)-1)
                    if autocorr[i-1] < autocorr[i] > autocorr[i+1]]
            
            if peaks:
                # Return the lag with highest autocorrelation
                return peaks[np.argmax([autocorr[i] for i in peaks])] + 1
            
            return None
        except:
            return None
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data with parallel processing and memory optimization."""
        print("\n🔍 Starting data analysis...")
        
        analysis = {
            'basic_stats': {},
            'data_quality': {},
            'patterns': {},
            'anomalies': {},
            'profile': {},
            'trends': {}
        }
        
        # Profile data
        analysis['profile'] = self._profile_data(df)
        
        # Analyze trends
        analysis['trends'] = self._analyze_trends(df)
        
        # Basic statistics with parallel processing
        print("📊 Calculating basic statistics...")
        analysis['basic_stats'] = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().to_dict(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Process numeric columns in parallel
        print("📈 Processing numeric columns...")
        for col in analysis['basic_stats']['numeric_columns']:
            try:
                stats = self._parallel_process(df[col], self._calculate_stats)
                analysis['basic_stats'][col] = {
                    k: float(v) for k, v in stats.items()
                }
            except Exception as e:
                print(f"⚠️ Error processing column {col}: {str(e)}")
                analysis['basic_stats'][col] = {
                    'error': str(e)
                }
        
        # Data quality assessment
        print("🔍 Assessing data quality...")
        try:
            analysis['data_quality'] = {
                'completeness': (1 - df.isnull().sum() / len(df)).to_dict(),
                'unique_values': df.nunique().to_dict(),
                'memory_usage': {col: str(df[col].memory_usage(deep=True) / 1024 / 1024) + ' MB'
                               for col in df.columns}
            }
        except Exception as e:
            print(f"⚠️ Error assessing data quality: {str(e)}")
            analysis['data_quality']['error'] = str(e)
        
        # Pattern detection with memory optimization
        if len(analysis['basic_stats']['numeric_columns']) >= 2:
            print("🔄 Detecting patterns...")
            try:
                numeric_df = df[analysis['basic_stats']['numeric_columns']]
                correlations = numeric_df.corr()
                analysis['patterns']['correlations'] = correlations.to_dict()
                
                # Find strong correlations
                strong_corr = []
                for i in range(len(correlations.columns)):
                    for j in range(i+1, len(correlations.columns)):
                        corr = correlations.iloc[i, j]
                        if abs(corr) > 0.7:  # Strong correlation threshold
                            strong_corr.append({
                                'variables': (correlations.columns[i], correlations.columns[j]),
                                'correlation': float(corr)
                            })
                analysis['patterns']['strong_correlations'] = strong_corr
                
                del numeric_df
                gc.collect()
            except Exception as e:
                print(f"⚠️ Error detecting patterns: {str(e)}")
                analysis['patterns']['error'] = str(e)
        
        # Anomaly detection with parallel processing
        print("🔍 Detecting anomalies...")
        for col in analysis['basic_stats']['numeric_columns']:
            try:
                outlier_stats = self._parallel_process(df[col], self._detect_outliers)
                analysis['anomalies'][col] = {
                    'outlier_count': int(outlier_stats['outlier_count']),
                    'outlier_percentage': float(outlier_stats['outlier_percentage']),
                    'min_outlier': outlier_stats['min_outlier'],
                    'max_outlier': outlier_stats['max_outlier']
                }
            except Exception as e:
                print(f"⚠️ Error detecting anomalies in {col}: {str(e)}")
                analysis['anomalies'][col] = {'error': str(e)}
        
        print("✅ Analysis complete!")
        return analysis
    
    def compare_datasets(self, file1_path: Union[str, Path], file2_path: Union[str, Path]) -> Dict:
        """Compare two datasets and identify key differences"""
        # Load both datasets
        df1 = self.load_data(file1_path)
        df2 = self.load_data(file2_path)
        
        comparison = {
            'structure': self._compare_structure(df1, df2),
            'statistics': self._compare_statistics(df1, df2),
            'distributions': self._compare_distributions(df1, df2),
            'changes': self._identify_changes(df1, df2)
        }
        
        return comparison
    
    def _compare_structure(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """Compare structural differences between datasets"""
        return {
            'columns': {
                'only_in_first': list(set(df1.columns) - set(df2.columns)),
                'only_in_second': list(set(df2.columns) - set(df1.columns)),
                'common': list(set(df1.columns) & set(df2.columns))
            },
            'row_counts': {
                'first': len(df1),
                'second': len(df2),
                'difference': len(df1) - len(df2)
            },
            'dtypes': {
                'first': {col: str(dtype) for col, dtype in df1.dtypes.items()},
                'second': {col: str(dtype) for col, dtype in df2.dtypes.items()}
            }
        }
    
    def _compare_statistics(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """Compare statistical measures between datasets"""
        common_cols = list(set(df1.columns) & set(df2.columns))
        stats = {}
        
        for col in common_cols:
            try:
                # Convert to numeric, coercing errors to NaN
                series1 = pd.to_numeric(df1[col], errors='coerce')
                series2 = pd.to_numeric(df2[col], errors='coerce')
                
                # Only compute stats if we have numeric values
                if not series1.isna().all() and not series2.isna().all():
                    stats[col] = {
                        'mean_diff': float(series2.mean() - series1.mean()),
                        'std_diff': float(series2.std() - series1.std()),
                        'min_diff': float(series2.min() - series1.min()),
                        'max_diff': float(series2.max() - series1.max()),
                        'null_counts': {
                            'first': int(series1.isna().sum()),
                            'second': int(series2.isna().sum())
                        }
                    }
            except Exception as e:
                stats[col] = {'error': str(e)}
        
        return stats
    
    def _compare_distributions(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """Compare value distributions between datasets"""
        common_cols = list(set(df1.columns) & set(df2.columns))
        distributions = {}
        
        for col in common_cols:
            try:
                if df1[col].dtype == df2[col].dtype:
                    # Try to convert to numeric
                    try:
                        series1 = pd.to_numeric(df1[col], errors='coerce')
                        series2 = pd.to_numeric(df2[col], errors='coerce')
                        
                        if not series1.isna().all() and not series2.isna().all():
                            # For numeric columns, compare histograms
                            hist1, bins1 = np.histogram(series1.dropna(), bins=20)
                            hist2, bins2 = np.histogram(series2.dropna(), bins=20)
                            
                            distributions[col] = {
                                'type': 'numeric',
                                'histogram_diff': [float(x) for x in (hist2 - hist1)],
                                'bins': [float(x) for x in bins1]
                            }
                        else:
                            # Handle as categorical if numeric conversion results in all NaN
                            vc1 = df1[col].value_counts(normalize=True)
                            vc2 = df2[col].value_counts(normalize=True)
                            distributions[col] = self._compare_categorical(vc1, vc2)
                    except:
                        # For categorical columns, compare value counts
                        vc1 = df1[col].value_counts(normalize=True)
                        vc2 = df2[col].value_counts(normalize=True)
                        distributions[col] = self._compare_categorical(vc1, vc2)
            except Exception as e:
                distributions[col] = {'error': str(e)}
        
        return distributions
    
    def _compare_categorical(self, vc1: pd.Series, vc2: pd.Series) -> Dict:
        """Compare categorical value distributions"""
        # Get all unique values
        all_values = list(set(vc1.index) | set(vc2.index))
        
        return {
            'type': 'categorical',
            'values': {
                str(value): {
                    'first': float(vc1.get(value, 0)),
                    'second': float(vc2.get(value, 0)),
                    'difference': float(vc2.get(value, 0) - vc1.get(value, 0))
                }
                for value in all_values
            }
        }
    
    def _identify_changes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """Identify significant changes between datasets"""
        changes = {
            'new_values': {},
            'removed_values': {},
            'significant_changes': {}
        }
        
        common_cols = list(set(df1.columns) & set(df2.columns))
        
        for col in common_cols:
            try:
                # Try to convert to numeric
                series1 = pd.to_numeric(df1[col], errors='coerce')
                series2 = pd.to_numeric(df2[col], errors='coerce')
                
                if not series1.isna().all() and not series2.isna().all():
                    # For numeric columns, identify significant changes
                    mean1 = series1.mean()
                    mean2 = series2.mean()
                    if mean1 != 0:  # Avoid division by zero
                        mean_change = (mean2 - mean1) / abs(mean1) * 100
                        if abs(mean_change) > 10:  # 10% threshold
                            changes['significant_changes'][col] = {
                                'percent_change': float(mean_change),
                                'absolute_change': float(mean2 - mean1)
                            }
                else:
                    # For categorical columns, identify new and removed values
                    values1 = set(df1[col].dropna().unique())
                    values2 = set(df2[col].dropna().unique())
                    
                    new_values = values2 - values1
                    removed_values = values1 - values2
                    
                    if new_values:
                        changes['new_values'][col] = [str(x) for x in new_values]
                    if removed_values:
                        changes['removed_values'][col] = [str(x) for x in removed_values]
            except Exception as e:
                changes[col] = {'error': str(e)}
        
        return changes
    
    def generate_report(self, analysis: Dict[str, Any], output_file: str) -> str:
        """Generate a JSON report with error handling."""
        try:
            # Ensure the output directory exists
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add metadata to the report
            report = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'comparison' if 'dataset1_name' in analysis else 'single_dataset',
                'results': self._convert_to_serializable(analysis)
            }
            
            # Convert the entire report to serializable format
            serializable_report = self._convert_to_serializable(report)
            
            # Verify JSON serialization before writing
            try:
                json.dumps(serializable_report)
            except Exception as e:
                print(f"⚠️ JSON serialization verification failed: {str(e)}")
                # Additional conversion for problematic values
                serializable_report = {
                    k: str(v) if not isinstance(v, (dict, list, int, float, str, bool, type(None)))
                    else v for k, v in serializable_report.items()
                }
            
            # Save analysis to JSON file
            with open(output_file, 'w') as f:
                json.dump(serializable_report, f, indent=2)
            
            return output_file
        except Exception as e:
            print(f"⚠️ Error generating report: {str(e)}")
            # Save to backup location with minimal data
            backup_file = str(output_file) + '.backup'
            backup_report = {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'partial_results': str(analysis)
            }
            with open(backup_file, 'w') as f:
                json.dump(backup_report, f, indent=2)
            return backup_file
    
    def export_to_power_bi(self, analysis_results: Dict, output_dir: str) -> Dict:
        """Export analysis results in Power BI compatible format
        Returns paths to exported files and metadata for Power BI integration"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_files = {}

        try:
            # Export basic statistics
            if 'basic_stats' in analysis_results:
                stats_df = pd.DataFrame(analysis_results['basic_stats']['numeric'])
                stats_file = output_dir / 'basic_stats.csv'
                stats_df.to_csv(stats_file)
                export_files['basic_stats'] = str(stats_file)

            # Export data quality metrics
            if 'data_quality' in analysis_results:
                quality_metrics = {
                    'missing_values': pd.Series(analysis_results['data_quality'].get('missing_values', {})),
                    'completeness': pd.Series(analysis_results['data_quality'].get('completeness', {}))
                }
                quality_df = pd.DataFrame(quality_metrics)
                quality_file = output_dir / 'data_quality.csv'
                quality_df.to_csv(quality_file)
                export_files['data_quality'] = str(quality_file)

            # Export correlation matrix
            if 'correlations' in analysis_results:
                if 'correlation_matrix' in analysis_results['correlations']:
                    corr_df = pd.DataFrame(analysis_results['correlations']['correlation_matrix'])
                    corr_file = output_dir / 'correlations.csv'
                    corr_df.to_csv(corr_file)
                    export_files['correlations'] = str(corr_file)

            # Create Power BI template
            template = {
                'version': '1.0',
                'datasets': list(export_files.keys()),
                'relationships': [
                    {
                        'name': 'Stats to Quality',
                        'fromTable': 'basic_stats',
                        'fromColumn': 'column',
                        'toTable': 'data_quality',
                        'toColumn': 'column'
                    }
                ],
                'suggested_visualizations': [
                    {
                        'type': 'matrix',
                        'dataset': 'correlations',
                        'title': 'Correlation Matrix'
                    },
                    {
                        'type': 'column_chart',
                        'dataset': 'data_quality',
                        'title': 'Data Completeness'
                    }
                ]
            }

            # Save template
            template_file = output_dir / 'power_bi_template.json'
            with open(template_file, 'w') as f:
                json.dump(template, f, indent=2)
            export_files['template'] = str(template_file)

            return export_files

        except Exception as e:
            raise Exception(f"Error exporting to Power BI format: {str(e)}")

    def export_to_tableau(self, analysis_results: Dict, output_dir: str) -> Dict:
        """Export analysis results in Tableau compatible format
        Returns paths to exported files and metadata for Tableau integration"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_files = {}

        try:
            # Export basic statistics with Tableau-specific formatting
            if 'basic_stats' in analysis_results:
                stats_df = pd.DataFrame(analysis_results['basic_stats']['numeric'])
                # Reshape for Tableau's preferred format
                stats_df = stats_df.reset_index().melt(id_vars=['index'])
                stats_df.columns = ['Metric', 'Column', 'Value']
                stats_file = output_dir / 'tableau_stats.csv'
                stats_df.to_csv(stats_file, index=False)
                export_files['basic_stats'] = str(stats_file)

            # Export data quality metrics in Tableau format
            if 'data_quality' in analysis_results:
                quality_data = []
                for metric, values in analysis_results['data_quality'].items():
                    if isinstance(values, dict):
                        for col, val in values.items():
                            quality_data.append({
                                'Metric': metric,
                                'Column': col,
                                'Value': val
                            })
                quality_df = pd.DataFrame(quality_data)
                quality_file = output_dir / 'tableau_quality.csv'
                quality_df.to_csv(quality_file, index=False)
                export_files['data_quality'] = str(quality_file)

            # Export correlation matrix in Tableau format
            if 'correlations' in analysis_results:
                if 'correlation_matrix' in analysis_results['correlations']:
                    corr_df = pd.DataFrame(analysis_results['correlations']['correlation_matrix'])
                    # Reshape for Tableau's preferred format
                    corr_df = corr_df.reset_index().melt(id_vars=['index'])
                    corr_df.columns = ['Variable1', 'Variable2', 'Correlation']
                    corr_file = output_dir / 'tableau_correlations.csv'
                    corr_df.to_csv(corr_file, index=False)
                    export_files['correlations'] = str(corr_file)

            # Create Tableau workbook template
            workbook_template = {
                'version': '2020.3',
                'worksheets': [
                    {
                        'name': 'Correlation Matrix',
                        'datasource': 'correlations',
                        'visualization': 'heat_map',
                        'columns': ['Variable1', 'Variable2'],
                        'measures': ['Correlation'],
                        'color_encoding': 'Correlation'
                    },
                    {
                        'name': 'Data Quality Overview',
                        'datasource': 'data_quality',
                        'visualization': 'bar_chart',
                        'columns': ['Column'],
                        'measures': ['Value'],
                        'color_encoding': 'Metric'
                    }
                ],
                'dashboard': {
                    'name': 'Data Analysis Overview',
                    'layout': 'grid',
                    'sheets': ['Correlation Matrix', 'Data Quality Overview']
                }
            }

            # Save template
            template_file = output_dir / 'tableau_template.json'
            with open(template_file, 'w') as f:
                json.dump(workbook_template, f, indent=2)
            export_files['template'] = str(template_file)

            return export_files

        except Exception as e:
            raise Exception(f"Error exporting to Tableau format: {str(e)}")

    def visualize_in_power_bi(self, analysis_results: Dict, workspace_name: Optional[str] = None) -> str:
        """
        Create and publish a Power BI report directly from analysis results
        Returns the URL of the published report or path to local visualization
        """
        try:
            # Check if Power BI credentials are available
            if not os.getenv('POWERBI_CLIENT_ID') or not os.getenv('POWERBI_CLIENT_SECRET'):
                return self._create_local_visualization(analysis_results, 'power_bi')
            
            # Initialize Power BI credentials
            credential = InteractiveBrowserCredential()
            
            # Convert analysis results to DataFrames
            dataframes = {}
            
            # Basic stats
            if 'basic_stats' in analysis_results:
                stats_df = pd.DataFrame(analysis_results['basic_stats'])
                dataframes['basic_stats'] = stats_df
            
            # Data quality
            if 'data_quality' in analysis_results:
                quality_df = pd.DataFrame(analysis_results['data_quality'])
                dataframes['data_quality'] = quality_df
            
            # Create visualizations using Power BI's QuickVisualize
            report = QuickVisualize(
                credential=credential,
                dataset_create_config={
                    'defaultMode': 'Push',
                    'enableRls': False
                }
            )
            
            # Add datasets
            for name, df in dataframes.items():
                report.add_dataset(df, name)
            
            # Create visualizations
            if 'data_quality' in dataframes:
                report.create_visual(
                    'data_quality',
                    visual_type='column_chart',
                    title='Data Completeness'
                )
            
            # Publish report
            if workspace_name:
                report_url = report.publish(workspace_name)
            else:
                report_url = report.publish("My Workspace")
            
            return report_url
            
        except Exception as e:
            print(f"⚠️ Power BI visualization error: {str(e)}")
            return self._create_local_visualization(analysis_results, 'power_bi')

    def visualize_in_tableau(self, analysis_results: Dict, 
                           project_name: str,
                           site_name: Optional[str] = None) -> str:
        """
        Create and publish a Tableau workbook directly from analysis results
        Returns the URL of the published workbook or path to local visualization
        """
        try:
            # Check if Tableau credentials are available
            if not os.getenv('TABLEAU_TOKEN_NAME') or not os.getenv('TABLEAU_TOKEN'):
                return self._create_local_visualization(analysis_results, 'tableau')
            
            # Initialize Tableau Server client
            tableau_auth = TSC.PersonalAccessTokenAuth(
                token_name=os.getenv('TABLEAU_TOKEN_NAME'),
                personal_access_token=os.getenv('TABLEAU_TOKEN'),
                site_id=site_name or ''
            )
            server = TSC.Server(os.getenv('TABLEAU_SERVER_URL'))
            
            # Sign in to server
            with server.auth.sign_in(tableau_auth):
                # Convert analysis results to DataFrames
                dataframes = self._prepare_visualization_data(analysis_results)
                
                # Create new project if it doesn't exist
                project = next((proj for proj in TSC.Pager(server.projects) 
                              if proj.name == project_name), None)
                if not project:
                    project = TSC.ProjectItem(project_name)
                    project = server.projects.create(project)
                
                # Create and publish workbook
                workbook = TSC.WorkbookItem(project_id=project.id)
                workbook.name = f"Data Analysis {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                
                # Create temporary files for the data
                temp_files = {}
                for name, df in dataframes.items():
                    temp_file = f"temp_{name}.hyper"
                    df.to_csv(temp_file, index=False)
                    temp_files[name] = temp_file
                
                try:
                    # Publish workbook with data sources
                    published_workbook = server.workbooks.publish(
                        workbook,
                        temp_files['stats'],
                        'Overwrite',
                        hidden_views=['_temp_view']
                    )
                    
                    # Get workbook URL
                    workbook_url = f"{os.getenv('TABLEAU_SERVER_URL')}/#/site/{site_name}/views/{published_workbook.id}"
                    
                    return workbook_url
                    
                finally:
                    # Clean up temporary files
                    for temp_file in temp_files.values():
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            
        except Exception as e:
            print(f"⚠️ Tableau visualization error: {str(e)}")
            return self._create_local_visualization(analysis_results, 'tableau')

    def _create_local_visualization(self, analysis_results: Dict, viz_type: str) -> str:
        """Create local visualizations using plotly"""
        try:
            # Create output directory
            output_dir = Path('../output/visualizations')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create visualizations based on analysis results
            figs = []
            
            # Data quality visualization
            if 'data_quality' in analysis_results:
                quality_data = analysis_results['data_quality'].get('completeness', {})
                if quality_data:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(quality_data.keys()),
                            y=list(quality_data.values()),
                            name='Completeness'
                        )
                    ])
                    fig.update_layout(
                        title='Data Completeness by Column',
                        xaxis_title='Column',
                        yaxis_title='Completeness Ratio',
                        template='plotly_white'
                    )
                    figs.append(('completeness', fig))
            
            # Patterns visualization
            if 'patterns' in analysis_results:
                correlations = analysis_results['patterns'].get('correlations', {})
                if correlations:
                    corr_df = pd.DataFrame(correlations)
                    fig = go.Figure(data=[
                        go.Heatmap(
                            z=corr_df.values,
                            x=corr_df.columns,
                            y=corr_df.index,
                            colorscale='RdBu'
                        )
                    ])
                    fig.update_layout(
                        title='Correlation Matrix',
                        template='plotly_white'
                    )
                    figs.append(('correlations', fig))
            
            # Data quality scores visualization
            if 'profile' in analysis_results:
                quality_scores = analysis_results['profile'].get('data_quality_score', {})
                if quality_scores:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(quality_scores.keys()),
                            y=list(quality_scores.values()),
                            marker_color='lightblue'
                        )
                    ])
                    fig.update_layout(
                        title='Data Quality Scores by Column',
                        xaxis_title='Column',
                        yaxis_title='Quality Score',
                        template='plotly_white',
                        showlegend=False
                    )
                    figs.append(('quality_scores', fig))
            
            # Distribution visualization for numeric columns
            if 'profile' in analysis_results:
                column_profiles = analysis_results['profile'].get('column_profiles', {})
                numeric_cols = [
                    col for col, profile in column_profiles.items()
                    if profile.get('distribution_type', '') != 'error'
                ]
                
                if numeric_cols:
                    dist_data = []
                    for col in numeric_cols:
                        dist_data.append({
                            'column': col,
                            'distribution': column_profiles[col]['distribution_type']
                        })
                    
                    dist_df = pd.DataFrame(dist_data)
                    fig = go.Figure(data=[
                        go.Bar(
                            x=dist_df['column'],
                            y=[1] * len(dist_df),
                            text=dist_df['distribution'],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title='Distribution Types by Column',
                        xaxis_title='Column',
                        yaxis_showticklabels=False,
                        template='plotly_white',
                        showlegend=False
                    )
                    figs.append(('distributions', fig))
            
            # Pattern detection visualization
            if 'profile' in analysis_results:
                column_profiles = analysis_results['profile'].get('column_profiles', {})
                pattern_data = []
                
                for col, profile in column_profiles.items():
                    if 'common_patterns' in profile:
                        for pattern, count in profile['common_patterns'].items():
                            pattern_data.append({
                                'column': col,
                                'pattern': pattern,
                                'count': count
                            })
                
                if pattern_data:
                    pattern_df = pd.DataFrame(pattern_data)
                    fig = go.Figure(data=[
                        go.Bar(
                            x=pattern_df['column'],
                            y=pattern_df['count'],
                            name=pattern_df['pattern']
                        )
                    ])
                    fig.update_layout(
                        title='Detected Patterns by Column',
                        xaxis_title='Column',
                        yaxis_title='Count',
                        template='plotly_white',
                        barmode='stack'
                    )
                    figs.append(('patterns', fig))
            
            # Save all figures
            viz_paths = []
            for name, fig in figs:
                output_file = output_dir / f'{name}_{viz_type}_{timestamp}.html'
                fig.write_html(str(output_file))
                viz_paths.append(str(output_file))
            
            # Create summary dashboard
            dashboard = go.Figure()
            
            # Add all figures to the dashboard
            for i, (name, fig) in enumerate(figs):
                for trace in fig.data:
                    dashboard.add_trace(trace)
                
                # Update layout for each section
                dashboard.update_layout(
                    title=f'Data Analysis Dashboard - {viz_type}',
                    template='plotly_white',
                    height=300 * len(figs),  # Adjust height based on number of visualizations
                    grid={'rows': len(figs), 'columns': 1, 'pattern': 'independent'},
                    showlegend=True
                )
            
            # Save dashboard
            dashboard_file = output_dir / f'dashboard_{viz_type}_{timestamp}.html'
            dashboard.write_html(str(dashboard_file))
            viz_paths.append(str(dashboard_file))
            
            return '\n'.join(viz_paths)
            
        except Exception as e:
            print(f"⚠️ Local visualization error: {str(e)}")
            return "Failed to create local visualizations"

    def _prepare_visualization_data(self, analysis_results: Dict) -> Dict[str, pd.DataFrame]:
        """Prepare data for visualization"""
        dataframes = {}
        
        # Basic stats
        if 'basic_stats' in analysis_results:
            stats_data = []
            for col, stats in analysis_results['basic_stats'].items():
                if isinstance(stats, dict) and not col.startswith('_'):
                    stats['column'] = col
                    stats_data.append(stats)
            if stats_data:
                dataframes['stats'] = pd.DataFrame(stats_data)
        
        # Data quality
        if 'data_quality' in analysis_results:
            quality_data = []
            for metric, values in analysis_results['data_quality'].items():
                if isinstance(values, dict):
                    for col, val in values.items():
                        quality_data.append({
                            'metric': metric,
                            'column': col,
                            'value': val
                        })
            if quality_data:
                dataframes['quality'] = pd.DataFrame(quality_data)
        
        # Patterns
        if 'patterns' in analysis_results:
            if 'correlations' in analysis_results['patterns']:
                corr_df = pd.DataFrame(analysis_results['patterns']['correlations'])
                dataframes['correlations'] = corr_df
        
        return dataframes

if __name__ == "__main__":
    # Example usage
    analyzer = CompanyDataAnalyzer(processing_mode="Memory Efficient")
    
    # Analyze a dataset
    df = analyzer.load_data("path_to_dataset.csv")
    analysis = analyzer.analyze_data(df)
    
    # Visualize in Power BI
    power_bi_url = analyzer.visualize_in_power_bi(
        analysis,
        workspace_name="My Analytics Workspace"
    )
    print(f"Power BI Report URL: {power_bi_url}")
    
    # Visualize in Tableau
    tableau_url = analyzer.visualize_in_tableau(
        analysis,
        project_name="Data Analysis",
        site_name="my_site"
    )
    print(f"Tableau Workbook URL: {tableau_url}")
    
    # Compare two datasets
    comparison = analyzer.compare_datasets(
        "path_to_first_dataset.csv",
        "path_to_second_dataset.csv"
    )
    
    # Generate and save report
    report = analyzer.generate_report(
        comparison,
        output_file="comparison_report.json"
    ) 