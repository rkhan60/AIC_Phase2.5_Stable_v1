import pandas as pd
import torch
from typing import Dict, List, Optional, Union
import json
import numpy as np
from pathlib import Path
import dask.dataframe as dd
import dask.array as da

class CompanyDataProcessor:
    """Processes company data for AI consulting analysis"""
    def __init__(self):
        self.required_fields = {
            'financial': ['revenue', 'costs', 'profit_margins', 'growth_rate'],
            'operational': ['employees', 'locations', 'capacity_utilization'],
            'market': ['market_share', 'competitors', 'target_segments'],
            'strategic': ['current_initiatives', 'challenges', 'objectives']
        }
        
    def validate_data(self, data: Union[Dict, List[Dict]]) -> bool:
        """Validate if all required fields are present"""
        if isinstance(data, dict) and 'datasets' in data:
            # Handle multiple datasets
            for dataset in data['datasets']:
                if not self._validate_single_dataset(dataset):
                    return False
            return True
        else:
            # Single dataset
            return self._validate_single_dataset(data)
    
    def _validate_single_dataset(self, data: Dict) -> bool:
        """Validate a single dataset"""
        try:
            for category, fields in self.required_fields.items():
                if category not in data:
                    print(f"Missing category: {category}")
                    return False
                for field in fields:
                    if field not in data[category]:
                        print(f"Missing field in {category}: {field}")
                        return False
            return True
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return False
    
    def process_financial_data(self, financial_data: Dict) -> torch.Tensor:
        """Convert financial data to tensor format"""
        try:
            financial_metrics = []
            for field in self.required_fields['financial']:
                value = financial_data.get(field, 0)
                if isinstance(value, str):
                    try:
                        value = float(value.replace(',', '').replace('$', ''))
                    except ValueError:
                        value = 0
                financial_metrics.append(value)
            return torch.tensor(financial_metrics, dtype=torch.float32)
        except Exception as e:
            print(f"Error processing financial data: {str(e)}")
            return torch.zeros(len(self.required_fields['financial']), dtype=torch.float32)
    
    def process_market_data(self, market_data: Dict) -> torch.Tensor:
        """Process market-related information"""
        market_metrics = []
        # Convert market share to float
        market_share = float(str(market_data['market_share']).rstrip('%')) / 100
        market_metrics.append(market_share)
        
        # Convert competitors to embedding
        competitor_count = len(market_data['competitors'])
        market_metrics.append(competitor_count)
        
        # Convert target segments to embedding
        segment_count = len(market_data['target_segments'])
        market_metrics.append(segment_count)
        
        return torch.tensor(market_metrics, dtype=torch.float32)
    
    def process_company_data(self, file_path: str) -> Dict[str, torch.Tensor]:
        """Process complete company data file"""
        try:
            # Read data file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Validate data
            if not self.validate_data(data):
                raise ValueError("Invalid data format")
            
            # Process each category
            processed_data = {
                'financial': self.process_financial_data(data['financial']),
                'market': self.process_market_data(data['market']),
                'operational': torch.tensor([
                    data['operational']['employees'],
                    len(data['operational']['locations']),
                    float(str(data['operational']['capacity_utilization']).rstrip('%')) / 100
                ], dtype=torch.float32),
                'strategic': self.process_strategic_data(data['strategic'])
            }
            
            return processed_data
            
        except Exception as e:
            raise Exception(f"Error processing company data: {str(e)}")
    
    def process_strategic_data(self, strategic_data: Dict) -> torch.Tensor:
        """Process strategic information"""
        # Convert initiatives to count
        initiative_count = len(strategic_data['current_initiatives'])
        
        # Convert challenges to count
        challenge_count = len(strategic_data['challenges'])
        
        # Convert objectives to count
        objective_count = len(strategic_data['objectives'])
        
        return torch.tensor([initiative_count, challenge_count, objective_count], dtype=torch.float32)
    
    def get_data_template(self) -> Dict:
        """Return template for required company data"""
        return {
            'financial': {field: '' for field in self.required_fields['financial']},
            'operational': {field: '' for field in self.required_fields['operational']},
            'market': {field: '' for field in self.required_fields['market']},
            'strategic': {field: '' for field in self.required_fields['strategic']}
        }
    
    def process_large_dataset(self, data: Union[Dict, List[Dict]], processing_mode: str = "Standard") -> Dict:
        """Process large datasets efficiently"""
        try:
            if isinstance(data, dict) and 'datasets' in data:
                # Multiple datasets
                processed_data = []
                for dataset in data['datasets']:
                    processed = self._process_single_dataset(dataset, processing_mode)
                    processed_data.append(processed)
                return self._combine_processed_data(processed_data)
            else:
                # Single dataset
                return self._process_single_dataset(data, processing_mode)
        except Exception as e:
            print(f"Error processing large dataset: {str(e)}")
            return {}
    
    def _process_single_dataset(self, data: Dict, processing_mode: str) -> Dict:
        """Process a single dataset based on processing mode"""
        try:
            if processing_mode == "Memory Efficient":
                # Use Dask for memory-efficient processing
                df = dd.from_pandas(pd.DataFrame([data]), npartitions=1)
                processed = df.map_partitions(self._process_partition).compute()
                return processed.to_dict('records')[0]
            elif processing_mode == "High Performance":
                # Use parallel processing with larger chunks
                return self._high_performance_processing(data)
            else:
                # Standard processing
                return self._standard_processing(data)
        except Exception as e:
            print(f"Error in single dataset processing: {str(e)}")
            return {}
    
    def _process_partition(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a partition of data"""
        # Process each category
        for category in self.required_fields.keys():
            if category in df.columns:
                df[f"{category}_processed"] = df[category].apply(self._process_category)
        return df
    
    def _process_category(self, category_data: Dict) -> Dict:
        """Process a single category of data"""
        processed = {}
        for field, value in category_data.items():
            if isinstance(value, (int, float)):
                processed[field] = value
            elif isinstance(value, str):
                try:
                    processed[field] = float(value.replace(',', '').replace('$', ''))
                except ValueError:
                    processed[field] = value
            elif isinstance(value, list):
                processed[field] = value
            else:
                processed[field] = str(value)
        return processed
    
    def _high_performance_processing(self, data: Dict) -> Dict:
        """High-performance processing using numpy arrays"""
        processed = {}
        for category, fields in self.required_fields.items():
            if category in data:
                category_data = data[category]
                if isinstance(category_data, dict):
                    # Convert numerical data to numpy arrays for faster processing
                    numerical_fields = {
                        field: value for field, value in category_data.items()
                        if isinstance(value, (int, float))
                    }
                    if numerical_fields:
                        arr = np.array(list(numerical_fields.values()))
                        # Apply any necessary transformations
                        processed[category] = {
                            field: value for field, value in zip(numerical_fields.keys(), arr)
                        }
                    # Keep non-numerical data as is
                    processed[category].update({
                        field: value for field, value in category_data.items()
                        if not isinstance(value, (int, float))
                    })
        return processed
    
    def _standard_processing(self, data: Dict) -> Dict:
        """Standard processing for moderate-sized data"""
        processed = {}
        for category, fields in self.required_fields.items():
            if category in data:
                processed[category] = self._process_category(data[category])
        return processed
    
    def _combine_processed_data(self, processed_data: List[Dict]) -> Dict:
        """Combine multiple processed datasets"""
        combined = {
            'financial': {},
            'operational': {},
            'market': {},
            'strategic': {}
        }
        
        # Combine numerical data
        for category in self.required_fields.keys():
            numerical_fields = []
            non_numerical_fields = {}
            
            for data in processed_data:
                if category in data:
                    for field, value in data[category].items():
                        if isinstance(value, (int, float)):
                            numerical_fields.append(value)
                        else:
                            if field not in non_numerical_fields:
                                non_numerical_fields[field] = []
                            non_numerical_fields[field].append(value)
            
            # Average numerical fields
            if numerical_fields:
                combined[category]['average'] = np.mean(numerical_fields)
                combined[category]['std'] = np.std(numerical_fields)
                combined[category]['min'] = np.min(numerical_fields)
                combined[category]['max'] = np.max(numerical_fields)
            
            # Combine non-numerical fields
            for field, values in non_numerical_fields.items():
                combined[category][field] = list(set(
                    item for sublist in values 
                    for item in (sublist if isinstance(sublist, list) else [sublist])
                )) 