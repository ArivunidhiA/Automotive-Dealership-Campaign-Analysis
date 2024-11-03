{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Campaign Analysis and Customer Segmentation\n",
    "\n",
    "This notebook demonstrates the analysis of automotive dealership campaign data using the project's core functionality."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from src.data_processing import DataProcessor\n",
    "from src.campaign_analysis import CampaignAnalyzer\n",
    "from src.predictive_models import PredictiveModel\n",
    "\n",
    "# Set up plotting\n",
    "plt.style.use('seaborn')\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Initialize data processor\n",
    "processor = DataProcessor('data/')\n",
    "\n",
    "# Load and process data\n",
    "sales_data = processor.load_sales_data()\n",
    "customer_metrics = processor.calculate_customer_metrics(sales_data)\n",
    "segmented_customers = processor.segment_customers(customer_metrics)\n",
    "\n",
    "print(\"Data Shape:\", sales_data.shape)\n",
    "print(\"\\nCustomer Segments Distribution:\")\n",
    "print(segmented_customers['segment'].value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Analyze campaign performance\n",
    "analyzer = CampaignAnalyzer()\n",
    "campaign_metrics = analyzer.calculate_campaign_metrics(sales_data)\n",
    "\n",
    "# Visualize campaign performance\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.scatterplot(\n",
    "    data=campaign_metrics,\n",
    "    x='response_rate',\n",
    "    y='avg_roi',\n",
    "    size='total_revenue',\n",
    "    sizes=(100, 1000),\n",
    "    alpha=0.6\n",
    ")\n",
    "plt.title('Campaign Performance Overview')\n",
    "plt.xlabel('Response Rate')\n",
    "plt.ylabel('Average ROI')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Train predictive model\n",
    "model = PredictiveModel()\n",
    "X, y = model.prepare_features(segmented_customers)\n",
    "results = model.train_model(X, y)\n",
    "\n",
    "# Plot feature importance\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(\n",
    "    data=results['feature_importance'],\n",
    "    x='importance',\n",
    "    y='feature'\n",
    ")\n",
    "plt.title('Feature Importance in Predicting Campaign Response')\n",
    "plt.show()\n",
    "\n",
    "print(\"\\nModel Performance:\")\n",
    "print(results['classification_report'])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
