# Directory structure
.
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_campaign_analysis.ipynb
│   ├── 03_predictive_modeling.ipynb
│   └── 04_financial_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── campaign_analysis.py
│   ├── customer_segmentation.py
│   ├── predictive_models.py
│   └── visualization.py
├── tests/
│   ├── __init__.py
│   └── test_data_processing.py
├── reports/
│   └── README.md
└── .gitignore

# Contents of README.md
# Customer Retention and Sales Performance Model for Automotive Dealership Campaigns

## Project Overview
This project analyzes customer engagement and retention across various marketing campaigns for automotive dealerships. It uses real-world automotive sales data to identify high-value customer segments, predict campaign outcomes, and recommend strategies for enhancing sales performance and customer loyalty.

## Data Sources
- Automotive Sales Dataset from Kaggle's "Car Sales in the United States" collection
- Customer Demographics and Campaign Response data
- Historical Campaign Performance Metrics

## Project Structure
- `data/`: Contains raw and processed data files
- `notebooks/`: Jupyter notebooks for analysis and visualization
- `src/`: Source code for data processing and analysis
- `tests/`: Unit tests
- `reports/`: Generated analysis reports and visualizations

## Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the required datasets and place them in the `data/` directory
4. Run the notebooks in sequential order

## Key Features
- Customer segmentation analysis
- Campaign effectiveness prediction
- Financial performance modeling
- Interactive dashboards
- Automated reporting

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- plotly
- jupyter

## License
MIT License

# Contents of requirements.txt
pandas==2.0.0
numpy==1.24.3
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.1
jupyter==1.0.0
pytest==7.3.1
python-dotenv==1.0.0

# Contents of src/data_processing.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class DataProcessor:
    """Handles data loading and preprocessing for automotive sales data."""
    
    def __init__(self, data_path: str):
        """Initialize with path to data directory."""
        self.data_path = data_path
        
    def load_sales_data(self) -> pd.DataFrame:
        """Load and clean sales data."""
        df = pd.read_csv(f"{self.data_path}/sales_data.csv")
        
        # Clean and standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        
        # Convert date columns to datetime
        date_columns = ['purchase_date', 'campaign_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        return df
    
    def calculate_customer_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate key customer metrics."""
        metrics = df.groupby('customer_id').agg({
            'purchase_amount': ['sum', 'mean', 'count'],
            'campaign_response': 'mean',
            'purchase_date': lambda x: (x.max() - x.min()).days
        }).reset_index()
        
        metrics.columns = ['customer_id', 'total_spend', 'avg_spend', 
                         'purchase_count', 'campaign_response_rate', 'customer_lifetime_days']
        
        # Calculate Customer Lifetime Value
        metrics['clv'] = metrics['total_spend'] * metrics['campaign_response_rate']
        
        return metrics

    def segment_customers(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Segment customers based on spending and engagement."""
        metrics['segment'] = pd.qcut(metrics['clv'], q=4, labels=[
            'Low Value', 'Medium Value', 'High Value', 'Premium'
        ])
        return metrics

# Contents of src/campaign_analysis.py
import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import StandardScaler

class CampaignAnalyzer:
    """Analyzes campaign performance and customer response patterns."""
    
    def calculate_campaign_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate key performance metrics for each campaign."""
        campaign_metrics = df.groupby('campaign_id').agg({
            'customer_id': 'count',
            'campaign_response': 'mean',
            'purchase_amount': ['sum', 'mean'],
            'roi': 'mean'
        }).reset_index()
        
        campaign_metrics.columns = [
            'campaign_id', 'total_customers', 'response_rate',
            'total_revenue', 'avg_purchase', 'avg_roi'
        ]
        
        return campaign_metrics
    
    def analyze_campaign_effectiveness(self, 
                                    df: pd.DataFrame,
                                    campaign_metrics: pd.DataFrame) -> Dict:
        """Analyze campaign effectiveness by customer segment."""
        segment_performance = df.groupby(['campaign_id', 'customer_segment']).agg({
            'campaign_response': 'mean',
            'purchase_amount': 'mean',
            'roi': 'mean'
        }).reset_index()
        
        best_campaigns = segment_performance.loc[
            segment_performance.groupby('customer_segment')['roi'].idxmax()
        ]
        
        return {
            'segment_performance': segment_performance,
            'best_campaigns': best_campaigns
        }

# Contents of src/predictive_models.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from typing import Dict, Tuple

class PredictiveModel:
    """Builds and evaluates predictive models for campaign response."""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.feature_importance = None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling."""
        # Select relevant features
        features = [
            'total_spend', 'purchase_count', 'avg_spend',
            'customer_lifetime_days', 'previous_campaign_response'
        ]
        
        X = df[features]
        y = df['campaign_response']
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the predictive model and return performance metrics."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': self.feature_importance
        }
    
    def predict_response(self, X_new: pd.DataFrame) -> np.ndarray:
        """Predict campaign response for new customers."""
        return self.model.predict_proba(X_new)

# Contents of .gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# VS Code
.vscode/

# Environment
.env
.venv
venv/
ENV/

# Data files
*.csv
*.xlsx
*.xls
*.db
*.sqlite

# OS specific
.DS_Store
Thumbs.db
