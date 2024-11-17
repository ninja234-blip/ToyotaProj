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
        yearly_stats = self.data.groupby('year').agg({
            self.mpg: 'mean',
            self.fc: 'mean',
            self.ghg: 'mean'
        }).reset_index()

        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=('Average MPG by Year',
                                         'Average Annual Fuel Cost by Year',
                                         'Average GHG Rating by Year'))

        fig.add_trace(go.Scatter(x=yearly_stats['year'], y=yearly_stats[mpg],
                               mode='lines+markers', name='MPG'),
                     row=1, col=1)
        
        fig.add_trace(go.Scatter(x=yearly_stats['year'], y=yearly_stats[fc],
                               mode='lines+markers', name='Fuel Cost'),
                     row=2, col=1)
        
        fig.add_trace(go.Scatter(x=yearly_stats['year'], y=yearly_stats[ghg],
                               mode='lines+markers', name='GHG Rating'),
                     row=3, col=1)

        fig.update_layout(height=800, showlegend=False)
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def create_model_comparison(self):
        """Create model comparison scatter plot"""
        model_stats = self.data.groupby(carline).agg({
            self.mpg: 'mean',
            self.fc: 'mean',
            self.ghg: 'mean',
        }).reset_index()

        fig = px.scatter(model_stats, x=mpg, y=fc,
                        size=ghg, hover_data=[carline],
                        title='Model Comparison: MPG vs Fuel Cost (size = GHG Rating)')
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    def get_summary_stats(self):
        """Get summary statistics"""
        return {
            'avg_mpg': round(self.data[mpg].mean(), 2),
            'avg_fuel_cost': round(self.data[fc].mean(), 2),
            'avg_ghg': round(self.data[ghg].mean(), 2),
            'total_models': len(self.data[carline].unique())
        }

    def predict_future(self, year, mpg, fuc):
        """Train model and make prediction"""
        features = ['year', mpg, fc]
        target = ghg
        
        X = self.data[features]
        y = self.data[target]
        
        model = LinearRegression()
        model.fit(X, y)
        
        prediction = model.predict([[year, mpg, fc]])
        return round(float(prediction[0]), 2)

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
    
    return jsonify({
        'yearly_trends': yearly_trends,
        'model_comparison': model_comparison,
        'summary_stats': summary_stats
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = analyzer.predict_future(
        int(data['year']),
        float(data[mpg]),
        float(data[fc])
    )
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)