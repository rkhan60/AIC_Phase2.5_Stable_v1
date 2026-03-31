from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from ..engine.agent_manager import AgentCapability, AgentTask, AgentResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AnalyticsAgent:
    """Lightweight base class for data-focused analytics agents.

    Replaces the nn.Module-based BusinessAgent, which is designed for
    strategic NLP tasks and requires mandatory constructor arguments that
    analytics agents don't need.
    """

    def __init__(self):
        self.capabilities: List[AgentCapability] = []

    def process_task(self, task: AgentTask) -> AgentResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EDAReport:
    """Exploratory Data Analysis report."""
    dataset_name: str
    summary_stats: Dict
    data_quality: Dict
    correlations: pd.DataFrame
    visualizations: List[Dict]
    insights: List[Dict]
    recommendations: List[str]


@dataclass
class CustomerModel:
    """Customer behaviour / prediction model container."""
    model_type: str          # 'churn' | 'segmentation' | 'ltv'
    features: List[str]
    model_params: Dict
    performance_metrics: Dict
    segments: Optional[Dict] = None
    predictions: Optional[pd.DataFrame] = None


@dataclass
class Dashboard:
    """BI Dashboard configuration."""
    name: str
    platform: str            # 'powerbi' | 'tableau'
    data_sources: List[Dict]
    refresh_schedule: str
    components: List[Dict]
    filters: List[Dict]
    data_pipeline: Dict


# ---------------------------------------------------------------------------
# EDAVisualizer
# ---------------------------------------------------------------------------

class EDAVisualizer(AnalyticsAgent):
    """Agent specialised in exploratory data analysis and visualisation."""

    def __init__(self):
        super().__init__()
        self.capabilities = [
            AgentCapability.DATA_ANALYSIS,
            AgentCapability.VISUALIZATION,
        ]
        self.viz_templates = {
            'distribution': self._create_distribution_plot,
            'correlation': self._create_correlation_plot,
            'temporal': self._create_temporal_plot,
            'categorical': self._create_categorical_plot,
            'geographic': self._create_geographic_plot,
        }
        self.analysis_modules = {
            'data_quality': self._analyze_data_quality,
            'correlations': self._analyze_correlations,
            'outliers': self._analyze_outliers,
            'patterns': self._analyze_patterns,
        }

    def process_task(self, task: AgentTask) -> AgentResult:
        try:
            if task.task_type == AgentCapability.DATA_ANALYSIS.value:
                eda_report = self._perform_eda(task.input_data)
                visualizations = self._create_visualizations(task.input_data, eda_report)
                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self.__class__.__name__,
                    output_data={
                        'eda_report': {
                            k: (v.to_dict() if isinstance(v, pd.DataFrame) else v)
                            for k, v in eda_report.__dict__.items()
                        },
                        'visualizations': visualizations,
                    },
                    insights=self._generate_insights(eda_report),
                    confidence=self._calculate_confidence(eda_report),
                    processing_time=0.0,
                    metadata={'dataset': eda_report.dataset_name},
                )
            raise ValueError(f"Unsupported task type: {task.task_type}")
        except Exception as exc:
            logger.error("Error in EDAVisualizer: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Core EDA
    # ------------------------------------------------------------------

    def _perform_eda(self, data: pd.DataFrame) -> EDAReport:
        summary_stats = {
            'basic_stats': data.describe(),
            'missing_values': data.isnull().sum(),
            'data_types': data.dtypes.to_dict(),
        }
        data_quality = self._analyze_data_quality(data)
        correlations = self._analyze_correlations(data)

        visualizations: List[Dict] = []
        for col in data.select_dtypes(include=[np.number]).columns:
            visualizations.append(self._create_distribution_plot(data[col]))

        insights = self._extract_insights(data, summary_stats, data_quality)

        return EDAReport(
            dataset_name=getattr(data, 'name', 'dataset'),
            summary_stats=summary_stats,
            data_quality=data_quality,
            correlations=correlations,
            visualizations=visualizations,
            insights=insights,
            recommendations=self._generate_recommendations(insights),
        )

    # ------------------------------------------------------------------
    # Analysis modules
    # ------------------------------------------------------------------

    def _analyze_data_quality(self, data: pd.DataFrame) -> Dict:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        missing_pct = (data.isnull().sum() / max(len(data), 1) * 100).to_dict()
        return {
            'missing_values': data.isnull().sum().to_dict(),
            'missing_pct': missing_pct,
            'duplicate_rows': int(data.duplicated().sum()),
            'duplicate_pct': float(data.duplicated().sum() / max(len(data), 1) * 100),
            'zero_variance_cols': [
                col for col in numeric_cols if data[col].std() == 0
            ],
            'constant_cols': [
                col for col in data.columns if data[col].nunique() <= 1
            ],
            'total_rows': len(data),
            'total_cols': len(data.columns),
            'memory_usage_mb': float(
                data.memory_usage(deep=True).sum() / 1024 ** 2
            ),
        }

    def _analyze_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return pd.DataFrame()
        return numeric_data.corr()

    def _analyze_outliers(self, data: pd.DataFrame) -> Dict:
        result: Dict = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (data[col] < lower) | (data[col] > upper)
            result[col] = {
                'count': int(mask.sum()),
                'pct': float(mask.mean() * 100),
                'lower_bound': float(lower),
                'upper_bound': float(upper),
            }
        return result

    def _analyze_patterns(self, data: pd.DataFrame) -> Dict:
        patterns: Dict = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            skew = float(data[col].skew())
            patterns[col] = {
                'skewness': skew,
                'distribution': (
                    'right-skewed' if skew > 0.5
                    else 'left-skewed' if skew < -0.5
                    else 'approximately-normal'
                ),
                'unique_values': int(data[col].nunique()),
                'range': float(data[col].max() - data[col].min()),
            }
        return patterns

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _create_distribution_plot(self, col_data: pd.Series) -> Dict:
        try:
            fig = px.histogram(col_data, title=f'Distribution: {col_data.name}')
            return {'type': 'distribution', 'column': str(col_data.name),
                    'data': fig.to_dict()}
        except Exception as exc:
            return {'type': 'distribution', 'column': str(col_data.name),
                    'error': str(exc)}

    def _create_correlation_plot(self, data: pd.DataFrame) -> Dict:
        try:
            corr = self._analyze_correlations(data)
            if corr.empty:
                return {'type': 'correlation', 'error': 'No numeric columns'}
            fig = px.imshow(corr, title='Correlation Matrix', text_auto='.2f')
            return {'type': 'correlation', 'data': fig.to_dict()}
        except Exception as exc:
            return {'type': 'correlation', 'error': str(exc)}

    def _create_temporal_plot(self, data: pd.DataFrame) -> Dict:
        try:
            date_cols = data.select_dtypes(include=['datetime64']).columns
            num_cols = data.select_dtypes(include=[np.number]).columns
            if date_cols.empty or num_cols.empty:
                return {'type': 'temporal', 'error': 'No datetime or numeric columns'}
            fig = px.line(data, x=date_cols[0], y=num_cols[0],
                          title=f'{num_cols[0]} over time')
            return {'type': 'temporal', 'data': fig.to_dict()}
        except Exception as exc:
            return {'type': 'temporal', 'error': str(exc)}

    def _create_categorical_plot(self, col_data: pd.Series) -> Dict:
        try:
            counts = col_data.value_counts().reset_index()
            counts.columns = ['value', 'count']
            fig = px.bar(counts, x='value', y='count',
                         title=f'Category Distribution: {col_data.name}')
            return {'type': 'categorical', 'column': str(col_data.name),
                    'data': fig.to_dict()}
        except Exception as exc:
            return {'type': 'categorical', 'column': str(col_data.name),
                    'error': str(exc)}

    def _create_geographic_plot(self, data: pd.DataFrame) -> Dict:
        # Placeholder: returns metadata only since location columns are dataset-specific
        return {
            'type': 'geographic',
            'note': 'Configure lat/lon columns before rendering',
            'data': None,
        }

    def _create_visualizations(
        self, data: pd.DataFrame, eda_report: EDAReport
    ) -> List[Dict]:
        vizs: List[Dict] = list(eda_report.visualizations)
        if not eda_report.correlations.empty:
            vizs.append(self._create_correlation_plot(data))
        for col in data.select_dtypes(include=['object', 'category']).columns[:3]:
            vizs.append(self._create_categorical_plot(data[col]))
        return vizs

    # ------------------------------------------------------------------
    # Insights & recommendations
    # ------------------------------------------------------------------

    def _extract_insights(
        self, data: pd.DataFrame, summary_stats: Dict, data_quality: Dict
    ) -> List[Dict]:
        insights: List[Dict] = []

        high_missing = {
            k: v for k, v in data_quality['missing_pct'].items() if v > 20
        }
        if high_missing:
            insights.append({
                'type': 'data_quality',
                'severity': 'high',
                'message': f'{len(high_missing)} column(s) have >20% missing values',
                'columns': list(high_missing.keys()),
            })

        if data_quality['duplicate_pct'] > 5:
            insights.append({
                'type': 'data_quality',
                'severity': 'medium',
                'message': (
                    f'{data_quality["duplicate_pct"]:.1f}% '
                    'duplicate rows detected'
                ),
            })

        outliers = self._analyze_outliers(data)
        high_outlier_cols = [c for c, v in outliers.items() if v['pct'] > 10]
        if high_outlier_cols:
            insights.append({
                'type': 'outliers',
                'severity': 'medium',
                'message': f'{len(high_outlier_cols)} column(s) with >10% outliers',
                'columns': high_outlier_cols,
            })

        return insights

    def _generate_insights(self, eda_report: EDAReport) -> Dict:
        basic = eda_report.summary_stats.get('basic_stats', pd.DataFrame())
        row_count = int(basic.shape[1]) if isinstance(basic, pd.DataFrame) else 0
        return {
            'summary': (
                f'Analysed {eda_report.data_quality.get("total_rows", 0)} rows, '
                f'{eda_report.data_quality.get("total_cols", 0)} columns'
            ),
            'key_findings': eda_report.insights[:5],
            'recommendations': eda_report.recommendations[:3],
        }

    def _generate_recommendations(self, insights: List[Dict]) -> List[str]:
        recs: List[str] = []
        for insight in insights:
            if insight['type'] == 'data_quality' and insight['severity'] == 'high':
                cols = ', '.join(insight.get('columns', []))
                recs.append(f'Impute or drop columns with high missingness: {cols}')
            elif insight['type'] == 'data_quality':
                recs.append('Remove or investigate duplicate rows')
            elif insight['type'] == 'outliers':
                cols = ', '.join(insight.get('columns', []))
                recs.append(f'Investigate outliers in: {cols}')
        if not recs:
            recs.append('Data quality is acceptable. Proceed with modelling.')
        return recs

    def _calculate_confidence(self, eda_report: EDAReport) -> float:
        missing_pcts = list(eda_report.data_quality.get('missing_pct', {}).values())
        avg_missing = float(np.mean(missing_pcts)) if missing_pcts else 0.0
        duplicate_pct = float(eda_report.data_quality.get('duplicate_pct', 0.0))
        quality_score = max(0.0, 1.0 - avg_missing / 100 - duplicate_pct / 200)
        return round(min(1.0, quality_score), 4)


# ---------------------------------------------------------------------------
# CustomerModeler
# ---------------------------------------------------------------------------

class CustomerModeler(AnalyticsAgent):
    """Agent specialised in customer behaviour modelling."""

    def __init__(self):
        super().__init__()
        self.capabilities = [
            AgentCapability.CUSTOMER_MODELING,
            AgentCapability.PREDICTIVE_ANALYTICS,
        ]
        self.model_configs = {
            'churn': self._get_churn_model_config(),
            'segmentation': self._get_segmentation_config(),
            'ltv': self._get_ltv_model_config(),
        }
        self.feature_pipelines = {
            'behavioral': self._create_behavioral_features,
            'transactional': self._create_transactional_features,
            'engagement': self._create_engagement_features,
        }

    def process_task(self, task: AgentTask) -> AgentResult:
        try:
            if task.task_type == AgentCapability.CUSTOMER_MODELING.value:
                model = self._build_customer_model(task.input_data)
                insights = self._analyze_model_results(model)
                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self.__class__.__name__,
                    output_data={'model': model.__dict__, 'insights': insights},
                    insights=self._generate_business_insights(model, insights),
                    confidence=self._calculate_confidence(model),
                    processing_time=0.0,
                    metadata={'model_type': model.model_type},
                )
            raise ValueError(f"Unsupported task type: {task.task_type}")
        except Exception as exc:
            logger.error("Error in CustomerModeler: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Model configurations
    # ------------------------------------------------------------------

    def _get_churn_model_config(self) -> Dict:
        return {
            'algorithm': 'gradient_boosting',
            'target_column': 'churned',
            'cv_folds': 3,
            'hyperparams': {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1},
        }

    def _get_segmentation_config(self) -> Dict:
        return {
            'algorithm': 'kmeans',
            'min_clusters': 2,
            'max_clusters': 8,
            'random_state': 42,
        }

    def _get_ltv_model_config(self) -> Dict:
        return {
            'algorithm': 'gradient_boosting_regressor',
            'target_column': 'ltv',
            'cv_folds': 3,
            'hyperparams': {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1},
        }

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _create_behavioral_features(self, data: pd.DataFrame) -> List[str]:
        keywords = ['session', 'visit', 'page', 'click', 'browse', 'view', 'search']
        return [
            col for col in data.select_dtypes(include=[np.number]).columns
            if any(kw in col.lower() for kw in keywords)
        ]

    def _create_transactional_features(self, data: pd.DataFrame) -> List[str]:
        keywords = ['purchase', 'order', 'spend', 'revenue', 'amount', 'transaction', 'price']
        return [
            col for col in data.select_dtypes(include=[np.number]).columns
            if any(kw in col.lower() for kw in keywords)
        ]

    def _create_engagement_features(self, data: pd.DataFrame) -> List[str]:
        keywords = ['email', 'open', 'click', 'login', 'active', 'engagement', 'nps']
        return [
            col for col in data.select_dtypes(include=[np.number]).columns
            if any(kw in col.lower() for kw in keywords)
        ]

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def _build_customer_model(self, data: Dict) -> CustomerModel:
        model_type = data.get('model_type', 'segmentation')
        config = self.model_configs[model_type]

        features: List[str] = []
        customer_data: pd.DataFrame = data.get('customer_data', pd.DataFrame())
        for pipeline_name, pipeline_fn in self.feature_pipelines.items():
            if pipeline_name in data.get('feature_sets', []):
                features.extend(pipeline_fn(customer_data))

        if model_type == 'segmentation':
            return self._build_segmentation_model(customer_data, features)
        elif model_type == 'churn':
            return self._build_churn_model(customer_data, features)
        else:
            return self._build_ltv_model(customer_data, features)

    def _build_segmentation_model(
        self, data: pd.DataFrame, features: List[str]
    ) -> CustomerModel:
        if not features:
            features = data.select_dtypes(include=[np.number]).columns.tolist()

        if not features or data.empty:
            return CustomerModel(
                model_type='segmentation',
                features=features,
                model_params={},
                performance_metrics={'error': 'No numeric features available'},
            )

        X = data[features].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        best_k = self._find_optimal_clusters(X_scaled)
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        segments = kmeans.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, segments))

        return CustomerModel(
            model_type='segmentation',
            features=features,
            model_params={'n_clusters': best_k},
            performance_metrics={'silhouette_score': sil},
            segments=self._analyze_segments(data, segments, features),
            predictions=pd.DataFrame({'segment': segments}),
        )

    def _build_churn_model(
        self, data: pd.DataFrame, features: List[str]
    ) -> CustomerModel:
        if not features:
            features = data.select_dtypes(include=[np.number]).columns.tolist()

        target_col = self.model_configs['churn']['target_column']
        if target_col not in data.columns or not features or data.empty:
            return CustomerModel(
                model_type='churn',
                features=features,
                model_params={},
                performance_metrics={'error': 'Missing target column or features'},
            )

        X = data[features].fillna(0)
        y = data[target_col]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = GradientBoostingClassifier(
            **self.model_configs['churn']['hyperparams']
        )
        cv_folds = min(self.model_configs['churn']['cv_folds'], int(y.value_counts().min()))
        scores = cross_val_score(clf, X_scaled, y, cv=cv_folds, scoring='roc_auc')

        return CustomerModel(
            model_type='churn',
            features=features,
            model_params={'algorithm': 'gradient_boosting'},
            performance_metrics={
                'cv_roc_auc': float(scores.mean()),
                'cv_std': float(scores.std()),
            },
        )

    def _build_ltv_model(
        self, data: pd.DataFrame, features: List[str]
    ) -> CustomerModel:
        if not features:
            features = data.select_dtypes(include=[np.number]).columns.tolist()

        target_col = self.model_configs['ltv']['target_column']
        if target_col not in data.columns or not features or data.empty:
            return CustomerModel(
                model_type='ltv',
                features=features,
                model_params={},
                performance_metrics={'error': 'Missing target column or features'},
            )

        X = data[features].fillna(0)
        y = data[target_col]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reg = GradientBoostingRegressor(
            **self.model_configs['ltv']['hyperparams']
        )
        cv_folds = self.model_configs['ltv']['cv_folds']
        scores = cross_val_score(reg, X_scaled, y, cv=cv_folds, scoring='r2')

        return CustomerModel(
            model_type='ltv',
            features=features,
            model_params={'algorithm': 'gradient_boosting_regressor'},
            performance_metrics={
                'cv_r2': float(scores.mean()),
                'cv_std': float(scores.std()),
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_optimal_clusters(self, X_scaled: np.ndarray) -> int:
        min_k = self.model_configs['segmentation']['min_clusters']
        max_k = self.model_configs['segmentation']['max_clusters']
        n_samples = len(X_scaled)

        if n_samples <= min_k:
            return min_k

        best_k, best_score = min_k, -1.0
        for k in range(min_k, min(max_k + 1, n_samples)):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = float(silhouette_score(X_scaled, labels))
            if score > best_score:
                best_score, best_k = score, k
        return best_k

    def _analyze_segments(
        self, data: pd.DataFrame, segments: np.ndarray, features: List[str]
    ) -> Dict:
        profiles: Dict = {}
        for seg_id in np.unique(segments):
            mask = segments == seg_id
            seg_data = data[mask]
            profile: Dict = {
                'size': int(mask.sum()),
                'pct': float(mask.mean() * 100),
            }
            for feat in features:
                if feat in data.columns:
                    profile[f'avg_{feat}'] = float(seg_data[feat].mean())
            profiles[f'segment_{int(seg_id)}'] = profile
        return profiles

    def _analyze_model_results(self, model: CustomerModel) -> Dict:
        return {
            'model_type': model.model_type,
            'feature_count': len(model.features),
            'performance': model.performance_metrics,
            'segments': model.segments,
        }

    def _generate_business_insights(
        self, model: CustomerModel, insights: Dict
    ) -> Dict:
        if model.model_type == 'churn':
            auc = model.performance_metrics.get('cv_roc_auc', 0.0)
            return {
                'summary': f'Churn model AUC: {auc:.2f}',
                'business_impact': (
                    'High' if auc > 0.8 else 'Medium' if auc > 0.65 else 'Low'
                ),
                'recommendations': [
                    'Focus retention spend on high-risk customer cohorts',
                    'Implement early-warning alerts at churn probability >0.6',
                ],
            }
        elif model.model_type == 'ltv':
            r2 = model.performance_metrics.get('cv_r2', 0.0)
            return {
                'summary': f'LTV model R²: {r2:.2f}',
                'business_impact': 'High' if r2 > 0.7 else 'Medium',
                'recommendations': [
                    'Prioritise high-LTV acquisition channels',
                    'Personalise offers by predicted LTV tier',
                ],
            }
        else:
            seg_count = len(model.segments) if model.segments else 0
            return {
                'summary': f'Identified {seg_count} customer segment(s)',
                'business_impact': 'Medium',
                'recommendations': [
                    'Tailor marketing campaigns by segment profile',
                    'Monitor segment migration on a monthly cadence',
                ],
            }

    def _calculate_confidence(self, model: CustomerModel) -> float:
        m = model.performance_metrics
        if 'cv_roc_auc' in m:
            return round(float(m['cv_roc_auc']), 4)
        if 'cv_r2' in m:
            return round(max(0.0, float(m['cv_r2'])), 4)
        if 'silhouette_score' in m:
            return round(float(m['silhouette_score']), 4)
        return 0.5


# ---------------------------------------------------------------------------
# BIEngineer
# ---------------------------------------------------------------------------

class BIEngineer(AnalyticsAgent):
    """Agent specialised in BI dashboard design and data pipeline planning."""

    def __init__(self):
        super().__init__()
        self.capabilities = [
            AgentCapability.BI_DEVELOPMENT,
            AgentCapability.DATA_PIPELINE,
        ]
        self.dashboard_templates = {
            'executive': self._get_executive_template(),
            'operational': self._get_operational_template(),
            'analytical': self._get_analytical_template(),
        }
        self.pipeline_configs = {
            'powerbi': self._get_powerbi_pipeline_config(),
            'tableau': self._get_tableau_pipeline_config(),
        }

    def process_task(self, task: AgentTask) -> AgentResult:
        try:
            if task.task_type == AgentCapability.BI_DEVELOPMENT.value:
                dashboard = self._design_dashboard(task.input_data)
                implementation_plan = self._create_implementation_plan(dashboard)
                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self.__class__.__name__,
                    output_data={
                        'dashboard': dashboard.__dict__,
                        'implementation_plan': implementation_plan,
                    },
                    insights=self._generate_dashboard_insights(dashboard),
                    confidence=self._calculate_confidence(dashboard),
                    processing_time=0.0,
                    metadata={'platform': dashboard.platform},
                )
            raise ValueError(f"Unsupported task type: {task.task_type}")
        except Exception as exc:
            logger.error("Error in BIEngineer: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Templates & configs
    # ------------------------------------------------------------------

    def _get_executive_template(self) -> Dict:
        return {
            'components': [
                {'type': 'kpi_cards', 'metrics': ['revenue', 'profit', 'growth']},
                {'type': 'trend_chart', 'period': 'monthly'},
                {'type': 'performance_gauge', 'target_metric': 'revenue'},
            ],
            'filters': [
                {'field': 'date_range', 'type': 'date'},
                {'field': 'region', 'type': 'dropdown'},
            ],
        }

    def _get_operational_template(self) -> Dict:
        return {
            'components': [
                {'type': 'real_time_metrics', 'refresh': '5min'},
                {'type': 'alert_panel', 'priority': 'high'},
                {'type': 'operational_table', 'rows': 50},
            ],
            'filters': [
                {'field': 'department', 'type': 'dropdown'},
                {'field': 'status', 'type': 'multiselect'},
            ],
        }

    def _get_analytical_template(self) -> Dict:
        return {
            'components': [
                {'type': 'scatter_matrix', 'max_vars': 6},
                {'type': 'correlation_heatmap'},
                {'type': 'distribution_comparison'},
                {'type': 'drill_down_table'},
            ],
            'filters': [
                {'field': 'dimension', 'type': 'dropdown'},
                {'field': 'metric', 'type': 'multiselect'},
                {'field': 'date_range', 'type': 'date'},
            ],
        }

    def _get_powerbi_pipeline_config(self) -> Dict:
        return {
            'connector': 'PowerBI',
            'refresh_type': 'scheduled',
            'gateway_required': True,
            'supported_sources': ['sql_server', 'sharepoint', 'excel', 'rest_api'],
            'row_level_security': True,
        }

    def _get_tableau_pipeline_config(self) -> Dict:
        return {
            'connector': 'Tableau',
            'refresh_type': 'extract',
            'server_required': True,
            'supported_sources': ['postgresql', 'mysql', 'excel', 'google_sheets'],
            'row_level_security': True,
        }

    # ------------------------------------------------------------------
    # Dashboard design
    # ------------------------------------------------------------------

    def _design_dashboard(self, requirements: Dict) -> Dashboard:
        platform = requirements.get('platform', 'powerbi')
        template_key = requirements.get('type', 'executive')
        template = dict(self.dashboard_templates.get(template_key,
                                                      self.dashboard_templates['executive']))
        return Dashboard(
            name=requirements.get('name', 'dashboard'),
            platform=platform,
            data_sources=requirements.get('data_sources', []),
            refresh_schedule=requirements.get('refresh_schedule', 'daily'),
            components=template['components'],
            filters=template['filters'],
            data_pipeline=self.pipeline_configs.get(platform, {}),
        )

    def _create_implementation_plan(self, dashboard: Dashboard) -> Dict:
        return {
            'phases': [
                {
                    'name': 'Data Pipeline Setup',
                    'duration': '1 week',
                    'tasks': [
                        'Configure data sources',
                        'Set up transformations',
                        'Implement refresh logic',
                    ],
                },
                {
                    'name': 'Dashboard Development',
                    'duration': '2 weeks',
                    'tasks': [
                        'Create base layout',
                        'Implement visualisations',
                        'Configure filters',
                    ],
                },
                {
                    'name': 'Testing & Deployment',
                    'duration': '1 week',
                    'tasks': [
                        'Test refresh pipeline',
                        'Validate calculations',
                        'Deploy to production',
                    ],
                },
            ],
            'requirements': {
                'tools': [dashboard.platform],
                'access': ['data_sources', 'deployment_environment'],
                'dependencies': ['data_warehouse_connection'],
            },
        }

    # ------------------------------------------------------------------
    # Insights & confidence
    # ------------------------------------------------------------------

    def _generate_dashboard_insights(self, dashboard: Dashboard) -> Dict:
        return {
            'summary': (
                f'{dashboard.name} dashboard on {dashboard.platform} '
                f'with {len(dashboard.components)} component(s)'
            ),
            'refresh_cadence': dashboard.refresh_schedule,
            'data_sources': len(dashboard.data_sources),
            'recommendations': [
                f'Enable row-level security on {dashboard.platform}',
                'Schedule refresh during off-peak hours',
                'Add automated data quality checks to the pipeline',
            ],
        }

    def _calculate_confidence(self, dashboard: Dashboard) -> float:
        # Confidence based on how well specified the dashboard requirements are
        score = 0.5
        if dashboard.data_sources:
            score += 0.2
        if dashboard.components:
            score += 0.2
        if dashboard.filters:
            score += 0.1
        return round(min(1.0, score), 4)
