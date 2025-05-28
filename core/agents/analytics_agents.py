from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from ..engine.business_agents import BusinessAgent
from ..engine.agent_manager import AgentCapability, AgentTask, AgentResult

logger = logging.getLogger(__name__)

@dataclass
class EDAReport:
    """Exploratory Data Analysis Report"""
    dataset_name: str
    summary_stats: Dict
    data_quality: Dict
    correlations: pd.DataFrame
    visualizations: List[Dict]
    insights: List[Dict]
    recommendations: List[str]

@dataclass
class CustomerModel:
    """Customer behavior and prediction models"""
    model_type: str  # churn/segmentation/ltv
    features: List[str]
    model_params: Dict
    performance_metrics: Dict
    segments: Optional[Dict] = None
    predictions: Optional[pd.DataFrame] = None

@dataclass
class Dashboard:
    """BI Dashboard configuration"""
    name: str
    platform: str  # powerbi/tableau
    data_sources: List[Dict]
    refresh_schedule: str
    components: List[Dict]
    filters: List[Dict]
    data_pipeline: Dict

class EDAVisualizer(BusinessAgent):
    """Agent specialized in exploratory data analysis and visualization"""
    
    def __init__(self):
        super().__init__()
        self.capabilities = [
            AgentCapability.DATA_ANALYSIS,
            AgentCapability.VISUALIZATION
        ]
        
        # Visualization templates
        self.viz_templates = {
            'distribution': self._create_distribution_plot,
            'correlation': self._create_correlation_plot,
            'temporal': self._create_temporal_plot,
            'categorical': self._create_categorical_plot,
            'geographic': self._create_geographic_plot
        }
        
        # Analysis modules
        self.analysis_modules = {
            'data_quality': self._analyze_data_quality,
            'correlations': self._analyze_correlations,
            'outliers': self._analyze_outliers,
            'patterns': self._analyze_patterns
        }
        
    def process_task(self, task: AgentTask) -> AgentResult:
        """Process EDA tasks"""
        try:
            if task.task_type == AgentCapability.DATA_ANALYSIS.value:
                # Perform EDA
                eda_report = self._perform_eda(task.input_data)
                visualizations = self._create_visualizations(task.input_data, eda_report)
                
                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self.__class__.__name__,
                    output_data={
                        'eda_report': eda_report.__dict__,
                        'visualizations': visualizations
                    },
                    insights=self._generate_insights(eda_report),
                    confidence=self._calculate_confidence(eda_report),
                    processing_time=0.0,
                    metadata={'dataset': eda_report.dataset_name}
                )
            else:
                raise ValueError(f"Unsupported task type: {task.task_type}")
                
        except Exception as e:
            logger.error(f"Error in EDAVisualizer: {str(e)}")
            raise
            
    def _perform_eda(self, data: pd.DataFrame) -> EDAReport:
        """Perform comprehensive EDA"""
        # Basic statistics
        summary_stats = {
            'basic_stats': data.describe(),
            'missing_values': data.isnull().sum(),
            'data_types': data.dtypes.to_dict()
        }
        
        # Data quality analysis
        data_quality = self.analysis_modules['data_quality'](data)
        
        # Correlation analysis
        correlations = self.analysis_modules['correlations'](data)
        
        # Generate visualizations
        visualizations = []
        for col in data.select_dtypes(include=[np.number]).columns:
            viz = self.viz_templates['distribution'](data[col])
            visualizations.append(viz)
            
        # Generate insights
        insights = self._extract_insights(data, summary_stats, data_quality)
        
        return EDAReport(
            dataset_name=getattr(data, 'name', 'dataset'),
            summary_stats=summary_stats,
            data_quality=data_quality,
            correlations=correlations,
            visualizations=visualizations,
            insights=insights,
            recommendations=self._generate_recommendations(insights)
        )

class CustomerModeler(BusinessAgent):
    """Agent specialized in customer behavior modeling"""
    
    def __init__(self):
        super().__init__()
        self.capabilities = [
            AgentCapability.CUSTOMER_MODELING,
            AgentCapability.PREDICTIVE_ANALYTICS
        ]
        
        # Model configurations
        self.model_configs = {
            'churn': self._get_churn_model_config(),
            'segmentation': self._get_segmentation_config(),
            'ltv': self._get_ltv_model_config()
        }
        
        # Feature engineering pipelines
        self.feature_pipelines = {
            'behavioral': self._create_behavioral_features,
            'transactional': self._create_transactional_features,
            'engagement': self._create_engagement_features
        }
        
    def process_task(self, task: AgentTask) -> AgentResult:
        """Process customer modeling tasks"""
        try:
            if task.task_type == AgentCapability.CUSTOMER_MODELING.value:
                # Build customer model
                model = self._build_customer_model(task.input_data)
                insights = self._analyze_model_results(model)
                
                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self.__class__.__name__,
                    output_data={
                        'model': model.__dict__,
                        'insights': insights
                    },
                    insights=self._generate_business_insights(model, insights),
                    confidence=self._calculate_confidence(model),
                    processing_time=0.0,
                    metadata={'model_type': model.model_type}
                )
            else:
                raise ValueError(f"Unsupported task type: {task.task_type}")
                
        except Exception as e:
            logger.error(f"Error in CustomerModeler: {str(e)}")
            raise
            
    def _build_customer_model(self, data: Dict) -> CustomerModel:
        """Build customer behavior model"""
        model_type = data.get('model_type', 'segmentation')
        config = self.model_configs[model_type].copy()
        
        # Feature engineering
        features = []
        for pipeline_name, pipeline in self.feature_pipelines.items():
            if pipeline_name in data.get('feature_sets', []):
                features.extend(pipeline(data['customer_data']))
                
        # Build model based on type
        if model_type == 'segmentation':
            model = self._build_segmentation_model(data['customer_data'], features)
        elif model_type == 'churn':
            model = self._build_churn_model(data['customer_data'], features)
        else:
            model = self._build_ltv_model(data['customer_data'], features)
            
        return model
        
    def _build_segmentation_model(self, data: pd.DataFrame, features: List[str]) -> CustomerModel:
        """Build customer segmentation model"""
        # Prepare data
        X = data[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal number of clusters
        best_k = self._find_optimal_clusters(X_scaled)
        
        # Build KMeans model
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        segments = kmeans.fit_predict(X_scaled)
        
        # Analyze segments
        segment_profiles = self._analyze_segments(data, segments, features)
        
        return CustomerModel(
            model_type='segmentation',
            features=features,
            model_params={'n_clusters': best_k},
            performance_metrics={'silhouette_score': silhouette_score(X_scaled, segments)},
            segments=segment_profiles,
            predictions=pd.DataFrame({'segment': segments})
        )

class BIEngineer(BusinessAgent):
    """Agent specialized in BI engineering and dashboard maintenance"""
    
    def __init__(self):
        super().__init__()
        self.capabilities = [
            AgentCapability.BI_DEVELOPMENT,
            AgentCapability.DATA_PIPELINE
        ]
        
        # Dashboard templates
        self.dashboard_templates = {
            'executive': self._get_executive_template(),
            'operational': self._get_operational_template(),
            'analytical': self._get_analytical_template()
        }
        
        # Data pipeline configurations
        self.pipeline_configs = {
            'powerbi': self._get_powerbi_pipeline_config(),
            'tableau': self._get_tableau_pipeline_config()
        }
        
    def process_task(self, task: AgentTask) -> AgentResult:
        """Process BI engineering tasks"""
        try:
            if task.task_type == AgentCapability.BI_DEVELOPMENT.value:
                # Design dashboard
                dashboard = self._design_dashboard(task.input_data)
                implementation_plan = self._create_implementation_plan(dashboard)
                
                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self.__class__.__name__,
                    output_data={
                        'dashboard': dashboard.__dict__,
                        'implementation_plan': implementation_plan
                    },
                    insights=self._generate_dashboard_insights(dashboard),
                    confidence=self._calculate_confidence(dashboard),
                    processing_time=0.0,
                    metadata={'platform': dashboard.platform}
                )
            else:
                raise ValueError(f"Unsupported task type: {task.task_type}")
                
        except Exception as e:
            logger.error(f"Error in BIEngineer: {str(e)}")
            raise
            
    def _design_dashboard(self, requirements: Dict) -> Dashboard:
        """Design BI dashboard"""
        platform = requirements.get('platform', 'powerbi')
        template = self.dashboard_templates[requirements.get('type', 'executive')].copy()
        
        return Dashboard(
            name=requirements.get('name', 'dashboard'),
            platform=platform,
            data_sources=requirements.get('data_sources', []),
            refresh_schedule=requirements.get('refresh_schedule', 'daily'),
            components=template['components'],
            filters=template['filters'],
            data_pipeline=self.pipeline_configs[platform]
        )
        
    def _create_implementation_plan(self, dashboard: Dashboard) -> Dict:
        """Create dashboard implementation plan"""
        return {
            'phases': [
                {
                    'name': 'Data Pipeline Setup',
                    'duration': '1 week',
                    'tasks': [
                        'Configure data sources',
                        'Set up transformations',
                        'Implement refresh logic'
                    ]
                },
                {
                    'name': 'Dashboard Development',
                    'duration': '2 weeks',
                    'tasks': [
                        'Create base layout',
                        'Implement visualizations',
                        'Configure filters'
                    ]
                },
                {
                    'name': 'Testing & Deployment',
                    'duration': '1 week',
                    'tasks': [
                        'Test refresh pipeline',
                        'Validate calculations',
                        'Deploy to production'
                    ]
                }
            ],
            'requirements': {
                'tools': [dashboard.platform],
                'access': ['data_sources', 'deployment_environment'],
                'dependencies': ['data_warehouse_connection']
            }
        } 