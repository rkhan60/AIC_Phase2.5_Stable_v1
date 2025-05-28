import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import os
import plotly.io as pio
from plotly.subplots import make_subplots

class BusinessAnalytics:
    def __init__(self, data_file=None):
        """Initialize with either real data or generate sample data"""
        if data_file and os.path.exists(data_file):
            self.data = pd.read_csv(data_file)
            self.data['Date'] = pd.to_datetime(self.data['Date'])
        else:
            self.data = self._generate_sample_data()
        
        self.data = self._prepare_data()
        os.makedirs('final_outputs', exist_ok=True)
    
    def _generate_sample_data(self):
        """Generate realistic sample data if no real data provided"""
        dates = pd.date_range(start='2020-01-01', end=datetime.now(), freq='M')
        base_revenue = 1000000
        seasonal_factor = np.sin(np.arange(len(dates)) * 2 * np.pi / 12) * 100000
        trend_factor = np.arange(len(dates)) * 20000
        
        data = pd.DataFrame({
            'Date': dates,
            'Revenue': base_revenue + seasonal_factor + trend_factor + np.random.normal(0, 50000, len(dates)),
            'Costs': (base_revenue * 0.7) + (seasonal_factor * 0.6) + (trend_factor * 0.65) + np.random.normal(0, 30000, len(dates))
        })
        return data
    
    def _prepare_data(self):
        """Prepare data and calculate key metrics"""
        data = self.data.copy()
        data['Profit'] = data['Revenue'] - data['Costs']
        data['Profit_Margin'] = (data['Profit'] / data['Revenue']) * 100
        data['Month'] = data['Date'].dt.strftime('%Y-%m')
        data['MonthNum'] = range(len(data))
        return data
    
    def predict_trends(self, months_ahead=6):
        """Predict future trends using linear regression"""
        X = self.data['MonthNum'].values.reshape(-1, 1)
        
        # Predict Revenue
        revenue_model = LinearRegression()
        revenue_model.fit(X, self.data['Revenue'])
        future_months = np.arange(len(self.data), len(self.data) + months_ahead).reshape(-1, 1)
        revenue_forecast = revenue_model.predict(future_months)
        
        # Predict Costs
        costs_model = LinearRegression()
        costs_model.fit(X, self.data['Costs'])
        costs_forecast = costs_model.predict(future_months)
        
        # Generate future dates
        last_date = self.data['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=31), 
                                   periods=months_ahead, freq='M')
        
        return future_dates, revenue_forecast, costs_forecast
    
    def generate_insights_report(self):
        """Generate comprehensive business insights with visualizations"""
        # Create main performance visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Revenue and Profit Trends with Predictions',
                'Monthly Growth Rates',
                'Profit Margin Analysis',
                'Revenue vs Costs Distribution'
            ),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                  [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Get predictions
        future_dates, revenue_forecast, costs_forecast = self.predict_trends()
        profit_forecast = revenue_forecast - costs_forecast
        
        # Plot 1: Revenue and Profit Trends with Predictions
        fig.add_trace(
            go.Scatter(x=self.data['Date'], y=self.data['Revenue'],
                      name='Actual Revenue', line=dict(color='#2ecc71')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=future_dates, y=revenue_forecast,
                      name='Predicted Revenue', line=dict(color='#2ecc71', dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.data['Date'], y=self.data['Profit'],
                      name='Actual Profit', line=dict(color='#e74c3c')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=future_dates, y=profit_forecast,
                      name='Predicted Profit', line=dict(color='#e74c3c', dash='dash')),
            row=1, col=1
        )
        
        # Plot 2: Monthly Growth Rates
        monthly_growth = self.data['Revenue'].pct_change() * 100
        fig.add_trace(
            go.Bar(x=self.data['Date'], y=monthly_growth,
                  name='Revenue Growth %', marker_color='#3498db'),
            row=1, col=2
        )
        
        # Plot 3: Profit Margin Analysis
        fig.add_trace(
            go.Scatter(x=self.data['Date'], y=self.data['Profit_Margin'],
                      name='Profit Margin %', line=dict(color='#9b59b6')),
            row=2, col=1
        )
        
        # Plot 4: Revenue vs Costs Distribution
        fig.add_trace(
            go.Scatter(x=self.data['Revenue'], y=self.data['Costs'],
                      mode='markers', name='Revenue vs Costs',
                      marker=dict(color='#f1c40f', size=8)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            title_text="Comprehensive Business Performance Analysis",
            showlegend=True,
            template='plotly_white'
        )
        
        # Save as PNG
        pio.write_image(fig, "final_outputs/business_analysis.png")
        
        # Generate text insights
        self._generate_text_insights()
    
    def _generate_text_insights(self):
        """Generate detailed text insights"""
        latest = self.data.iloc[-1]
        previous = self.data.iloc[-2]
        yoy = self.data.iloc[-13] if len(self.data) > 12 else self.data.iloc[0]
        
        # Calculate trend strengths
        X = self.data['MonthNum'].values.reshape(-1, 1)
        y_revenue = self.data['Revenue'].values
        revenue_r2 = r2_score(y_revenue, LinearRegression().fit(X, y_revenue).predict(X))
        
        insights = [
            "=== EXECUTIVE BUSINESS INSIGHTS SUMMARY ===\n",
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
            "\nCURRENT PERFORMANCE METRICS:",
            f"• Revenue: ${latest['Revenue']:,.2f}",
            f"• Profit: ${latest['Profit']:,.2f}",
            f"• Profit Margin: {latest['Profit_Margin']:.1f}%\n",
            "\nGROWTH ANALYSIS:",
            f"• Month-over-Month Revenue Growth: {((latest['Revenue']/previous['Revenue'])-1)*100:.1f}%",
            f"• Year-over-Year Revenue Growth: {((latest['Revenue']/yoy['Revenue'])-1)*100:.1f}%",
            f"• Trend Strength (R²): {revenue_r2:.2f}\n",
            "\nKEY FINDINGS:",
            "• " + self._get_trend_insight(),
            "• " + self._get_margin_insight(),
            "• " + self._get_prediction_insight()
        ]
        
        # Save insights to file
        with open('final_outputs/business_insights.txt', 'w') as f:
            f.write('\n'.join(insights))
    
    def _get_trend_insight(self):
        """Generate trend insight"""
        recent_trend = self.data['Revenue'].tail(3).pct_change().mean() * 100
        if recent_trend > 5:
            return "Strong positive growth trend observed in recent months"
        elif recent_trend > 0:
            return "Moderate positive growth trend observed"
        else:
            return "Declining trend observed - attention required"
    
    def _get_margin_insight(self):
        """Generate margin insight"""
        recent_margins = self.data['Profit_Margin'].tail(3).mean()
        if recent_margins > 25:
            return "Excellent profit margins maintained"
        elif recent_margins > 15:
            return "Healthy profit margins, but room for improvement"
        else:
            return "Profit margins below target - cost optimization recommended"
    
    def _get_prediction_insight(self):
        """Generate prediction insight"""
        _, revenue_forecast, costs_forecast = self.predict_trends(3)
        future_growth = (revenue_forecast[-1] - revenue_forecast[0]) / revenue_forecast[0] * 100
        if future_growth > 10:
            return "Strong growth projected for next quarter"
        elif future_growth > 0:
            return "Moderate growth projected for next quarter"
        else:
            return "Challenging conditions projected - strategic planning recommended"

def main():
    # Initialize analyzer (it will use sample data since no real data provided)
    analyzer = BusinessAnalytics()
    
    # Generate comprehensive report
    analyzer.generate_insights_report()
    
    print("\nAnalysis completed! Check the 'final_outputs' folder for:")
    print("1. business_analysis.png - Visual dashboard with all key metrics")
    print("2. business_insights.txt - Detailed business insights and recommendations")

if __name__ == "__main__":
    main() 