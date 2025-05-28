import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import os

class BusinessVisualizer:
    def __init__(self):
        self.data = self._load_or_generate_data()
        os.makedirs('visualizations', exist_ok=True)
    
    def _load_or_generate_data(self):
        """Generate sample business data"""
        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='D')
        
        # Create a comprehensive dataset
        data = pd.DataFrame({
            'date': dates,
            'revenue': np.random.normal(1000000, 100000, len(dates)) * (1 + np.arange(len(dates)) * 0.001),
            'customers': np.random.normal(1000, 100, len(dates)) * (1 + np.arange(len(dates)) * 0.0005),
            'satisfaction': np.random.normal(4.2, 0.3, len(dates)),
            'costs': np.random.normal(800000, 80000, len(dates)) * (1 + np.arange(len(dates)) * 0.0008)
        })
        
        # Add some derived metrics
        data['profit'] = data['revenue'] - data['costs']
        data['profit_margin'] = (data['profit'] / data['revenue']) * 100
        data['month'] = data['date'].dt.strftime('%Y-%m')
        
        return data
    
    def show_revenue_trends(self):
        """Generate revenue trend visualization"""
        monthly_data = self.data.groupby('month').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'customers': 'mean'
        }).reset_index()
        
        fig = go.Figure()
        
        # Revenue line
        fig.add_trace(go.Scatter(
            x=monthly_data['month'],
            y=monthly_data['revenue'],
            name='Revenue',
            line=dict(color='#2ecc71', width=3),
            hovertemplate='Month: %{x}<br>Revenue: $%{y:,.2f}<extra></extra>'
        ))
        
        # Profit line
        fig.add_trace(go.Scatter(
            x=monthly_data['month'],
            y=monthly_data['profit'],
            name='Profit',
            line=dict(color='#e74c3c', width=3),
            hovertemplate='Month: %{x}<br>Profit: $%{y:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Monthly Revenue and Profit Trends',
            xaxis_title='Month',
            yaxis_title='Amount ($)',
            template='plotly_white',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.write_html('visualizations/revenue_trends.html')
        return 'visualizations/revenue_trends.html'
    
    def show_customer_metrics(self):
        """Generate customer metrics dashboard"""
        monthly_data = self.data.groupby('month').agg({
            'customers': ['mean', 'max', 'min'],
            'satisfaction': 'mean'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Customer Growth', 'Customer Satisfaction Trend')
        )
        
        # Customer growth
        fig.add_trace(
            go.Scatter(
                x=monthly_data['month'],
                y=monthly_data['customers']['mean'],
                name='Average Customers',
                line=dict(color='#3498db', width=3),
                hovertemplate='Month: %{x}<br>Customers: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Satisfaction trend
        fig.add_trace(
            go.Scatter(
                x=monthly_data['month'],
                y=monthly_data['satisfaction']['mean'],
                name='Satisfaction Score',
                line=dict(color='#9b59b6', width=3),
                hovertemplate='Month: %{x}<br>Score: %{y:.2f}/5.0<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text='Customer Performance Metrics',
            showlegend=True,
            template='plotly_white'
        )
        
        fig.write_html('visualizations/customer_metrics.html')
        return 'visualizations/customer_metrics.html'
    
    def show_profitability_analysis(self):
        """Generate profitability analysis dashboard"""
        monthly_data = self.data.groupby('month').agg({
            'revenue': 'sum',
            'costs': 'sum',
            'profit_margin': 'mean'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Revenue vs Costs', 'Profit Margin Trend'),
            vertical_spacing=0.15
        )
        
        # Revenue vs Costs
        fig.add_trace(
            go.Bar(
                x=monthly_data['month'],
                y=monthly_data['revenue'],
                name='Revenue',
                marker_color='#2ecc71',
                hovertemplate='Month: %{x}<br>Revenue: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=monthly_data['month'],
                y=monthly_data['costs'],
                name='Costs',
                marker_color='#e74c3c',
                hovertemplate='Month: %{x}<br>Costs: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Profit Margin Trend
        fig.add_trace(
            go.Scatter(
                x=monthly_data['month'],
                y=monthly_data['profit_margin'],
                name='Profit Margin',
                line=dict(color='#f1c40f', width=3),
                hovertemplate='Month: %{x}<br>Margin: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            title_text='Profitability Analysis',
            showlegend=True,
            template='plotly_white',
            barmode='group'
        )
        
        fig.write_html('visualizations/profitability_analysis.html')
        return 'visualizations/profitability_analysis.html'

def main():
    # Create visualizer instance
    viz = BusinessVisualizer()
    
    # Generate all visualizations
    revenue_file = viz.show_revenue_trends()
    customer_file = viz.show_customer_metrics()
    profitability_file = viz.show_profitability_analysis()
    
    print("\nVisualizations have been generated!")
    print(f"\nYou can find the interactive dashboards here:")
    print(f"1. Revenue Trends: {revenue_file}")
    print(f"2. Customer Metrics: {customer_file}")
    print(f"3. Profitability Analysis: {profitability_file}")
    print("\nOpen these files in your web browser to view the interactive visualizations.")

if __name__ == "__main__":
    main() 