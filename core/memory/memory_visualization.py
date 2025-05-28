from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import threading
import queue
import time
import json

class MemoryVisualizer:
    """Visualization system for memory analytics"""
    
    def __init__(self, analytics_manager, update_interval: int = 5):
        self.analytics = analytics_manager
        self.update_interval = update_interval
        self.alert_queue = queue.Queue()
        self.dashboard_app = self._create_dashboard()
        
    def _create_dashboard(self) -> dash.Dash:
        """Create interactive dashboard"""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1('Memory System Analytics Dashboard'),
            
            # Real-time Metrics
            html.Div([
                html.H2('Real-time Metrics'),
                dcc.Graph(id='memory-metrics-graph'),
                dcc.Interval(
                    id='metrics-update',
                    interval=self.update_interval * 1000
                )
            ]),
            
            # Memory Distribution
            html.Div([
                html.H2('Memory Distribution'),
                dcc.Graph(id='memory-distribution'),
                dcc.Interval(
                    id='distribution-update',
                    interval=self.update_interval * 2000
                )
            ]),
            
            # Performance Trends
            html.Div([
                html.H2('Performance Trends'),
                dcc.Graph(id='performance-trends'),
                dcc.Dropdown(
                    id='trend-timeframe',
                    options=[
                        {'label': '1 Hour', 'value': '1H'},
                        {'label': '24 Hours', 'value': '24H'},
                        {'label': '7 Days', 'value': '7D'}
                    ],
                    value='24H'
                )
            ]),
            
            # Memory Usage Patterns
            html.Div([
                html.H2('Memory Usage Patterns'),
                dcc.Graph(id='usage-patterns'),
                dcc.Interval(
                    id='patterns-update',
                    interval=self.update_interval * 3000
                )
            ]),
            
            # Memory Age Analysis
            html.Div([
                html.H2('Memory Age Analysis'),
                dcc.Graph(id='age-analysis'),
                dcc.Interval(
                    id='age-update',
                    interval=self.update_interval * 3000
                )
            ]),
            
            # Export Controls
            html.Div([
                html.H2('Export Options'),
                html.Button('Export PDF Report', id='export-pdf'),
                html.Button('Export Excel Report', id='export-excel'),
                html.Button('Export JSON Data', id='export-json'),
                html.Div(id='export-status')
            ]),
            
            # Alerts Panel
            html.Div([
                html.H2('System Alerts'),
                html.Div(id='alerts-panel'),
                dcc.Interval(
                    id='alerts-update',
                    interval=self.update_interval * 1000
                )
            ])
        ])
        
        self._setup_callbacks(app)
        return app
        
    def _setup_callbacks(self, app: dash.Dash):
        """Setup dashboard callback functions"""
        
        @app.callback(
            Output('memory-metrics-graph', 'figure'),
            Input('metrics-update', 'n_intervals')
        )
        def update_metrics_graph(_):
            return self._create_metrics_visualization()
            
        @app.callback(
            Output('memory-distribution', 'figure'),
            Input('distribution-update', 'n_intervals')
        )
        def update_distribution(_):
            return self._create_distribution_visualization()
            
        @app.callback(
            Output('performance-trends', 'figure'),
            [Input('trend-timeframe', 'value')]
        )
        def update_trends(timeframe):
            return self._create_trends_visualization(timeframe)
            
        @app.callback(
            Output('usage-patterns', 'figure'),
            Input('patterns-update', 'n_intervals')
        )
        def update_usage_patterns(_):
            return self._create_usage_patterns_visualization()
            
        @app.callback(
            Output('age-analysis', 'figure'),
            Input('age-update', 'n_intervals')
        )
        def update_age_analysis(_):
            return self._create_age_analysis_visualization()
            
        @app.callback(
            Output('export-status', 'children'),
            [Input('export-pdf', 'n_clicks'),
             Input('export-excel', 'n_clicks'),
             Input('export-json', 'n_clicks')]
        )
        def handle_export(pdf_clicks, excel_clicks, json_clicks):
            ctx = dash.callback_context
            if not ctx.triggered:
                return ""
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'export-pdf':
                self.export_pdf_report()
                return "PDF report exported successfully!"
            elif button_id == 'export-excel':
                self.export_excel_report()
                return "Excel report exported successfully!"
            elif button_id == 'export-json':
                self.export_json_data()
                return "JSON data exported successfully!"
                
        @app.callback(
            Output('alerts-panel', 'children'),
            Input('alerts-update', 'n_intervals')
        )
        def update_alerts(_):
            return self._create_alerts_panel()
            
    def _create_metrics_visualization(self) -> go.Figure:
        """Create real-time metrics visualization"""
        metrics = pd.DataFrame(self.analytics.metrics_history)
        
        fig = go.Figure()
        
        # Add traces for key metrics
        fig.add_trace(go.Scatter(
            x=metrics['timestamp'],
            y=metrics['total_memories'],
            name='Total Memories'
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics['timestamp'],
            y=metrics['active_memories'],
            name='Active Memories'
        ))
        
        fig.add_trace(go.Scatter(
            x=metrics['timestamp'],
            y=metrics['cache_hit_rate'],
            name='Cache Hit Rate',
            yaxis='y2'
        ))
        
        # Update layout
        fig.update_layout(
            title='Memory System Metrics Over Time',
            xaxis_title='Time',
            yaxis_title='Count',
            yaxis2=dict(
                title='Rate',
                overlaying='y',
                side='right'
            )
        )
        
        return fig
        
    def _create_distribution_visualization(self) -> go.Figure:
        """Create memory distribution visualization"""
        stats = self.analytics.analyze_usage_patterns(None, None)  # Pass actual instances in production
        
        # Create sunburst chart
        data = {
            'labels': [],
            'parents': [],
            'values': []
        }
        
        # Add confidence distribution
        for conf_level, count in stats.confidence_distribution.items():
            data['labels'].append(f'Confidence: {conf_level}')
            data['parents'].append('Memories')
            data['values'].append(count)
            
        # Add age distribution
        for age_range, count in stats.memory_age_distribution.items():
            data['labels'].append(f'Age: {age_range}')
            data['parents'].append('Memories')
            data['values'].append(count)
            
        fig = go.Figure(go.Sunburst(
            labels=data['labels'],
            parents=data['parents'],
            values=data['values']
        ))
        
        fig.update_layout(title='Memory Distribution')
        return fig
        
    def _create_trends_visualization(self, timeframe: str) -> go.Figure:
        """Create performance trends visualization"""
        # Convert timeframe to timedelta
        if timeframe == '1H':
            period = timedelta(hours=1)
        elif timeframe == '24H':
            period = timedelta(days=1)
        else:
            period = timedelta(days=7)
            
        report = self.analytics.generate_analytics_report(period)
        
        # Create heatmap of performance trends
        trends_data = []
        for metric, trend in report['metrics_trends'].items():
            trends_data.append({
                'metric': metric,
                'trend': trend,
                'impact': abs(trend)
            })
            
        df = pd.DataFrame(trends_data)
        
        fig = go.Figure(go.Heatmap(
            z=df['trend'].values.reshape(1, -1),
            x=df['metric'],
            y=['Trend'],
            colorscale='RdYlGn'
        ))
        
        fig.update_layout(
            title=f'Performance Trends ({timeframe})',
            xaxis_title='Metrics',
            yaxis_title='Trend Direction'
        )
        
        return fig
        
    def _create_usage_patterns_visualization(self) -> go.Figure:
        """Create visualization of memory usage patterns"""
        stats = self.analytics.analyze_usage_patterns(None, None)
        
        # Create scatter plot matrix
        df = pd.DataFrame({
            'access_frequency': list(stats.access_frequency.values()),
            'confidence': list(stats.confidence_distribution.values()),
            'age': list(stats.memory_age_distribution.values())
        })
        
        fig = px.scatter_matrix(df)
        fig.update_layout(title='Memory Usage Pattern Correlations')
        return fig
        
    def _create_age_analysis_visualization(self) -> go.Figure:
        """Create visualization of memory age distribution"""
        stats = self.analytics.analyze_usage_patterns(None, None)
        
        # Create stacked bar chart
        categories = list(stats.memory_age_distribution.keys())
        values = list(stats.memory_age_distribution.values())
        confidence_values = list(stats.confidence_distribution.values())
        
        fig = go.Figure(data=[
            go.Bar(name='Age Distribution', x=categories, y=values),
            go.Bar(name='Confidence Distribution', x=categories, y=confidence_values)
        ])
        
        fig.update_layout(
            title='Memory Age and Confidence Analysis',
            barmode='stack'
        )
        return fig
        
    def _create_alerts_panel(self) -> html.Div:
        """Create alerts panel"""
        alerts = []
        while not self.alert_queue.empty():
            alert = self.alert_queue.get()
            alerts.append(html.Div([
                html.H4(alert['title']),
                html.P(alert['description']),
                html.P(f"Priority: {alert['priority']}")
            ], className=f"alert-{alert['priority']}"))
            
        return html.Div(alerts)
        
    def start_monitoring(self):
        """Start real-time monitoring thread"""
        self.monitoring_thread = threading.Thread(
            target=self._monitor_system,
            daemon=True
        )
        self.monitoring_thread.start()
        
    def _monitor_system(self):
        """Monitor system metrics and generate alerts"""
        while True:
            metrics = self.analytics.collect_metrics(None, None)  # Pass actual instances in production
            
            # Check for alert conditions
            if metrics.cache_hit_rate < 0.5:
                self.alert_queue.put({
                    'title': 'Low Cache Hit Rate',
                    'description': 'Cache performance has degraded significantly',
                    'priority': 'high'
                })
                
            if metrics.query_latency_ms > 200:
                self.alert_queue.put({
                    'title': 'High Query Latency',
                    'description': 'Memory queries are taking longer than expected',
                    'priority': 'high'
                })
                
            if metrics.avg_confidence < 0.6:
                self.alert_queue.put({
                    'title': 'Low Confidence Levels',
                    'description': 'Average memory confidence has dropped',
                    'priority': 'medium'
                })
                
            time.sleep(self.update_interval)
            
    def run_dashboard(self, host: str = 'localhost', port: int = 8050):
        """Run the interactive dashboard"""
        self.start_monitoring()
        self.dashboard_app.run_server(host=host, port=port)
        
    def generate_static_report(self, output_path: Path):
        """Generate static HTML report"""
        metrics_fig = self._create_metrics_visualization()
        distribution_fig = self._create_distribution_visualization()
        trends_fig = self._create_trends_visualization('7D')
        
        html_content = f"""
        <html>
        <head>
            <title>Memory Analytics Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        </head>
        <body>
            <h1>Memory System Analytics Report</h1>
            <div id="metrics">{metrics_fig.to_html()}</div>
            <div id="distribution">{distribution_fig.to_html()}</div>
            <div id="trends">{trends_fig.to_html()}</div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
    def export_pdf_report(self):
        """Export analytics report as PDF"""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        
        doc = SimpleDocTemplate("memory_analytics_report.pdf", pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        story.append(Paragraph("Memory Analytics Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Add metrics
        metrics = self.analytics.collect_metrics(None, None)
        story.append(Paragraph("Current Metrics:", styles['Heading1']))
        story.append(Paragraph(f"Total Memories: {metrics.total_memories}", styles['Normal']))
        story.append(Paragraph(f"Active Memories: {metrics.active_memories}", styles['Normal']))
        story.append(Paragraph(f"Cache Hit Rate: {metrics.cache_hit_rate:.2%}", styles['Normal']))
        
        doc.build(story)
        
    def export_excel_report(self):
        """Export analytics data as Excel"""
        writer = pd.ExcelWriter('memory_analytics.xlsx', engine='openpyxl')
        
        # Export metrics history
        metrics_df = pd.DataFrame(self.analytics.metrics_history)
        metrics_df.to_excel(writer, sheet_name='Metrics History')
        
        # Export usage patterns
        patterns_df = pd.DataFrame(self.analytics.usage_patterns)
        patterns_df.to_excel(writer, sheet_name='Usage Patterns')
        
        writer.save()
        
    def export_json_data(self):
        """Export analytics data as JSON"""
        export_data = {
            'metrics_history': self.analytics.metrics_history,
            'usage_patterns': self.analytics.usage_patterns
        }
        
        with open('memory_analytics.json', 'w') as f:
            json.dump(export_data, f, indent=2) 