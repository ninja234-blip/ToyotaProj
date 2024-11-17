# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from werkzeug.utils import secure_filename
import math

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

mpg = "comb_fe_(guide)_-_conventional_fuel"
fc = "annual_fuel1_cost_-_conventional_fuel"
ghg = "ghg_rating_(1-10_rating_on_label)"
carline = "carline"

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class ToyotaAnalyzer:
    def __init__(self):
        self.data = None
        self.model = None

    def load_data(self, file_paths):
        """Load and combine multiple Excel files"""
        dfs = []
        for file_path in file_paths:
            df = pd.read_excel(file_path, engine='openpyxl')
            year = int(os.path.basename(file_path).split('.')[0][3])
            if 'Year' not in df.columns:
                df['Year'] = year
            dfs.append(df)
        
        self.data = pd.concat(dfs, ignore_index=True)
        self.data.columns = self.data.columns.str.strip().str.replace(' ', '_').str.lower()
        self.preprocess_data()

    def preprocess_data(self):
        """Clean and prepare data"""
        if mpg in self.data.columns:
            self.data[mpg] = pd.to_numeric(self.data[mpg], errors='coerce')
        
        if fc in self.data.columns:
            self.data[fc] = self.data[fc].replace('[\$,]', '', regex=True).astype(float)
        
        if ghg in self.data.columns:
            self.data[ghg] = pd.to_numeric(self.data[ghg], errors='coerce')

        self.mpg = 'comb_fe_(guide)_-_conventional_fuel'  # Updated to match the preprocessed name
        self.fc = 'annual_fuel1_cost_-_conventional_fuel'
        self.ghg = 'ghg_rating_(1-10_rating_on_label)'

        print("Columns in DataFrame after preprocessing:", self.data.columns)

    def create_yearly_trends(self):
        """Create yearly trend plots"""
        # Create a figure with subplots for MPG, Fuel Cost, and GHG
        fig = make_subplots(rows=3, cols=1,
                            subplot_titles=('Fuel Economy',
                                            'Annual Fuel Cost',
                                            'GHG Rating'))

        # Scatter plot of individual model data for MPG
        fig.add_trace(go.Scatter(x=self.data['year'], 
                                y=self.data[self.mpg],
                                mode='markers',
                                name='Individual Models - MPG',
                                marker=dict(size=5, opacity=0.5)),
                    row=1, col=1)

        # Fuel Cost Individual Models
        fig.add_trace(go.Scatter(x=self.data['year'], 
                                y=self.data[self.fc],
                                mode='markers',
                                name='Individual Models - Fuel Cost',
                                marker=dict(size=5, opacity=0.5)),
                    row=2, col=1)

        # GHG Rating Individual Models
        fig.add_trace(go.Scatter(x=self.data['year'], 
                                y=self.data[self.ghg],
                                mode='markers',
                                name='Individual Models - GHG Rating',
                                marker=dict(size=5, opacity=0.5)),
                    row=3, col=1)

        # Update layout for better visibility
        fig.update_layout(height=800, showlegend=True)

        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def create_model_comparison(self):
        """Create model comparison scatter plot"""
        model_stats = self.data.groupby([carline, 'year']).agg({
            self.mpg: 'mean',
            self.fc: 'mean',
            self.ghg: 'mean',
        }).reset_index()

        fig = px.scatter(
        model_stats, 
        x=self.mpg, 
        y=self.fc,
        size=self.ghg, 
        color='year',  # Color by year
        hover_data=[carline],
        title='Model Comparison: Fuel Economy vs Fuel Cost (size = GHG Rating)'
    )
        
        fig.update_layout(title='',
                      xaxis_title='Fuel Economy',
                      yaxis_title='Annual Fuel Cost',
                      legend_title='Year',
                      boxmode='group')  # Grouping the boxes together for comparison
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def get_summary_stats(self):
        """Get summary statistics"""
        return {
            'avg_mpg': round(self.data[mpg].mean(), 2),
            'avg_fuel_cost': round(self.data[fc].mean(), 2),
            'avg_ghg': round(self.data[ghg].mean(), 2),
            'total_models': len(self.data[carline].unique())
        }
    
    #to display info for mean of median of years for files
    def get_average_of_medians(self):
        """Calculate the average of the medians of MPG for each year."""
        median_data = self.data.groupby('year')[self.mpg].median()
        return round(median_data.mean(), 2) if not median_data.empty else None

    def predict_future(self, future_year):
        """Predict future average fuel economy based on median of past years' values."""
        # Pre-processing to ensure years are ordered
        median_data = self.data.groupby('year')[self.mpg].median().reset_index()
        
        if future_year not in median_data['year'].values:
            median_data = median_data.set_index('year')
            
            # Interpolate to find the median value for the future_year
            median_data = median_data[['comb_fe_(guide)_-_conventional_fuel']]
            future_prediction = median_data.reindex(range(median_data.index.min(), future_year + 1)).interpolate(method='linear')
            
            if future_year in future_prediction.index:
                predicted_value = future_prediction.loc[future_year].values[0]
                return round(float(predicted_value), 2)
            else:
                raise ValueError(f"No data available to predict for the year {future_year}.")

        # If there is data for that year, use the median directly
        return round(predicted_value, 2)

# Initialize analyzer
analyzer = ToyotaAnalyzer()

@app.route('/')
def index():
    return render_template('toyota.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    file_paths = []
    
    for file in files:
        if file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            file_paths.append(filepath)
    
    analyzer.load_data(file_paths)
    
    # Clean up uploaded files
    for filepath in file_paths:
        os.remove(filepath)
    
    return jsonify({'message': 'Files processed successfully'})

@app.route('/get_plots')
def get_plots():
    if analyzer.data is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    yearly_trends = analyzer.create_yearly_trends()
    model_comparison = analyzer.create_model_comparison()
    summary_stats = analyzer.get_summary_stats()

    avg_of_medians = analyzer.get_average_of_medians()
    
    return jsonify({
        'yearly_trends': yearly_trends,
        'model_comparison': model_comparison,
        'summary_stats': summary_stats,
        'avg_of_medians': avg_of_medians
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Capture the future year from the user's request
    future_year = int(data['future_year'])

    years_difference = future_year - 2025
        
    initial_increment = 0.4
    total_increment = 0

    for i in range(1, years_difference+1):
        total_increment += initial_increment * math.sqrt(1 - (i-1)/(years_difference+1))


    # Call the prediction function
    try:
        prediction = round(analyzer.predict_future(future_year) + total_increment, 2)
        return jsonify({'predicted_mpg': prediction})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)