from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

from ..engine.business_agents import BusinessAgent
from ..engine.agent_manager import AgentCapability, AgentTask, AgentResult

logger = logging.getLogger(__name__)

@dataclass
class CleaningConfig:
    """Data cleaning configuration"""
    dataset_name: str
    cleaning_steps: List[str]
    column_types: Dict[str, str]
    missing_value_strategy: Dict[str, str]
    outlier_strategy: Dict[str, str]
    validation_rules: Dict[str, Any]
    output_format: str

@dataclass
class CleaningReport:
    """Data cleaning results and metrics"""
    original_shape: tuple
    cleaned_shape: tuple
    missing_values_handled: Dict[str, int]
    outliers_handled: Dict[str, int]
    validation_results: Dict[str, bool]
    column_statistics: Dict[str, Dict]
    quality_score: float

@dataclass
class LabelingProject:
    """Labeling project configuration"""
    name: str
    task_type: str  # classification/object_detection/segmentation/text_annotation
    data_type: str  # image/text/audio/video
    annotation_schema: Dict
    instructions: Dict
    quality_checks: List[Dict]
    export_format: str

@dataclass
class LabelingTemplate:
    """Annotation template configuration"""
    template_type: str
    ui_components: List[Dict]
    validation_rules: Dict
    hotkeys: Dict
    export_settings: Dict

class DataCleaner(BusinessAgent):
    """Agent specialized in data cleaning and preprocessing"""
    
    def __init__(self):
        super().__init__()
        self.capabilities = [
            AgentCapability.DATA_CLEANING,
            AgentCapability.DATA_VALIDATION
        ]
        
        # Cleaning strategies
        self.cleaning_strategies = {
            'missing_values': {
                'mean': self._impute_mean,
                'median': self._impute_median,
                'mode': self._impute_mode,
                'knn': self._impute_knn,
                'custom': self._impute_custom
            },
            'outliers': {
                'zscore': self._handle_outliers_zscore,
                'iqr': self._handle_outliers_iqr,
                'isolation_forest': self._handle_outliers_iforest
            },
            'formatting': {
                'dates': self._standardize_dates,
                'numbers': self._standardize_numbers,
                'text': self._standardize_text,
                'categories': self._standardize_categories
            }
        }
        
        # Validation rules
        self.validation_rules = {
            'type_check': self._validate_types,
            'range_check': self._validate_ranges,
            'uniqueness': self._validate_uniqueness,
            'consistency': self._validate_consistency,
            'custom': self._validate_custom
        }
        
    def process_task(self, task: AgentTask) -> AgentResult:
        """Process data cleaning tasks"""
        try:
            if task.task_type == AgentCapability.DATA_CLEANING.value:
                # Configure and execute cleaning
                config = self._create_cleaning_config(task.input_data)
                cleaned_data, report = self._clean_data(task.input_data['data'], config)
                
                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self.__class__.__name__,
                    output_data={
                        'cleaned_data': cleaned_data,
                        'cleaning_report': report.__dict__
                    },
                    insights=self._generate_cleaning_insights(report),
                    confidence=self._calculate_confidence(report),
                    processing_time=0.0,
                    metadata={'dataset': config.dataset_name}
                )
            else:
                raise ValueError(f"Unsupported task type: {task.task_type}")
                
        except Exception as e:
            logger.error(f"Error in DataCleaner: {str(e)}")
            raise
            
    def _create_cleaning_config(self, requirements: Dict) -> CleaningConfig:
        """Create data cleaning configuration"""
        return CleaningConfig(
            dataset_name=requirements.get('dataset_name', 'dataset'),
            cleaning_steps=requirements.get('cleaning_steps', ['missing_values', 'outliers', 'formatting']),
            column_types=requirements.get('column_types', {}),
            missing_value_strategy=requirements.get('missing_value_strategy', {}),
            outlier_strategy=requirements.get('outlier_strategy', {}),
            validation_rules=requirements.get('validation_rules', {}),
            output_format=requirements.get('output_format', 'csv')
        )
        
    def _clean_data(self, data: pd.DataFrame, config: CleaningConfig) -> tuple[pd.DataFrame, CleaningReport]:
        """Execute data cleaning pipeline"""
        original_shape = data.shape
        cleaned_data = data.copy()
        missing_values_handled = {}
        outliers_handled = {}
        
        # Handle missing values
        for col, strategy in config.missing_value_strategy.items():
            if strategy in self.cleaning_strategies['missing_values']:
                cleaned_data, handled = self.cleaning_strategies['missing_values'][strategy](cleaned_data, col)
                missing_values_handled[col] = handled
                
        # Handle outliers
        for col, strategy in config.outlier_strategy.items():
            if strategy in self.cleaning_strategies['outliers']:
                cleaned_data, handled = self.cleaning_strategies['outliers'][strategy](cleaned_data, col)
                outliers_handled[col] = handled
                
        # Apply formatting
        for step in config.cleaning_steps:
            if 'format' in step and step.split('_')[1] in self.cleaning_strategies['formatting']:
                cleaned_data = self.cleaning_strategies['formatting'][step.split('_')[1]](cleaned_data)
                
        # Validate data
        validation_results = {}
        for rule, params in config.validation_rules.items():
            if rule in self.validation_rules:
                validation_results[rule] = self.validation_rules[rule](cleaned_data, params)
                
        # Generate report
        report = CleaningReport(
            original_shape=original_shape,
            cleaned_shape=cleaned_data.shape,
            missing_values_handled=missing_values_handled,
            outliers_handled=outliers_handled,
            validation_results=validation_results,
            column_statistics=self._generate_column_statistics(cleaned_data),
            quality_score=self._calculate_quality_score(cleaned_data, validation_results)
        )
        
        return cleaned_data, report

class LabelAgent(BusinessAgent):
    """Agent specialized in setting up data labeling projects"""
    
    def __init__(self):
        super().__init__()
        self.capabilities = [
            AgentCapability.LABELING_SETUP,
            AgentCapability.ANNOTATION_MANAGEMENT
        ]
        
        # Labeling templates
        self.templates = {
            'image_classification': self._get_image_classification_template(),
            'object_detection': self._get_object_detection_template(),
            'text_classification': self._get_text_classification_template(),
            'named_entity': self._get_ner_template(),
            'audio_transcription': self._get_audio_transcription_template()
        }
        
        # Quality assurance configs
        self.qa_configs = {
            'consensus': self._setup_consensus_checks,
            'gold_standard': self._setup_gold_standard,
            'cross_validation': self._setup_cross_validation,
            'time_tracking': self._setup_time_tracking
        }
        
    def process_task(self, task: AgentTask) -> AgentResult:
        """Process labeling setup tasks"""
        try:
            if task.task_type == AgentCapability.LABELING_SETUP.value:
                # Configure labeling project
                project = self._create_labeling_project(task.input_data)
                template = self._create_labeling_template(project)
                
                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self.__class__.__name__,
                    output_data={
                        'project_config': project.__dict__,
                        'template': template.__dict__
                    },
                    insights=self._generate_setup_insights(project),
                    confidence=self._calculate_confidence(project),
                    processing_time=0.0,
                    metadata={'project_type': project.task_type}
                )
            else:
                raise ValueError(f"Unsupported task type: {task.task_type}")
                
        except Exception as e:
            logger.error(f"Error in LabelAgent: {str(e)}")
            raise
            
    def _create_labeling_project(self, requirements: Dict) -> LabelingProject:
        """Create labeling project configuration"""
        return LabelingProject(
            name=requirements.get('name', 'labeling_project'),
            task_type=requirements.get('task_type', 'classification'),
            data_type=requirements.get('data_type', 'image'),
            annotation_schema=self._create_annotation_schema(requirements),
            instructions=self._create_labeling_instructions(requirements),
            quality_checks=self._setup_quality_checks(requirements),
            export_format=requirements.get('export_format', 'json')
        )
        
    def _create_labeling_template(self, project: LabelingProject) -> LabelingTemplate:
        """Create annotation template based on project requirements"""
        template_type = f"{project.data_type}_{project.task_type}"
        base_template = self.templates.get(template_type, self.templates['image_classification'])()
        
        return LabelingTemplate(
            template_type=template_type,
            ui_components=self._customize_ui_components(base_template, project),
            validation_rules=self._create_validation_rules(project),
            hotkeys=self._setup_hotkeys(project),
            export_settings=self._create_export_settings(project)
        )
        
    def _create_annotation_schema(self, requirements: Dict) -> Dict:
        """Create annotation schema based on task type"""
        task_type = requirements.get('task_type', 'classification')
        if task_type == 'classification':
            return {
                'labels': requirements.get('labels', []),
                'allow_multiple': requirements.get('allow_multiple', False),
                'hierarchical': requirements.get('hierarchical', False)
            }
        elif task_type == 'object_detection':
            return {
                'objects': requirements.get('objects', []),
                'attributes': requirements.get('attributes', {}),
                'relationships': requirements.get('relationships', [])
            }
        elif task_type == 'segmentation':
            return {
                'classes': requirements.get('classes', []),
                'overlay_options': requirements.get('overlay_options', {}),
                'brush_sizes': requirements.get('brush_sizes', [])
            }
        else:
            return {
                'entities': requirements.get('entities', []),
                'relations': requirements.get('relations', []),
                'attributes': requirements.get('attributes', {})
            } 