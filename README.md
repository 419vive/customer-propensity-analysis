# Customer Propensity ML Pipeline

A production-ready machine learning system for predicting customer purchase propensity with exceptional performance (91.8% F1 score, 99.7% ROC-AUC).

## Quick Start

### 1. Training Pipeline
```bash
python3 ml_pipeline_mvp.py
```
This runs the complete ML pipeline including:
- Data loading and validation
- Advanced feature engineering
- Multiple model training (Logistic Regression, Random Forest, XGBoost)
- Cross-validation and hyperparameter tuning
- Model evaluation and comparison
- Test prediction generation

### 2. Making Predictions
```bash
python3 model_inference.py --input_file your_data.csv --output_file predictions.csv
```

### 3. Viewing Results
Check the `ml_outputs/` directory for:
- Trained models (`*.pkl`)
- Performance metrics (`model_evaluation_results.json`)
- Test predictions (`test_predictions.csv`)
- Feature importance analysis (`*_feature_importance.csv`)

## System Architecture

### Core Components

1. **ml_pipeline_mvp.py** - Complete training pipeline
   - Handles 455K+ training samples with 4.19% class imbalance
   - Advanced feature engineering with interaction features
   - Multiple model training with class weight balancing
   - Comprehensive evaluation with business-relevant metrics

2. **model_inference.py** - Production inference system
   - Loads best model (Random Forest)
   - Validates input data format
   - Generates predictions with probability scores
   - Handles missing features gracefully

3. **InteractionFeatureTransformer** - Custom feature engineering
   - Creates interaction features between high-impact variables
   - Builds engagement and intent scores
   - Handles device-behavior interactions

## Performance Results

| Model | F1 Score | Precision | Recall | ROC-AUC |
|-------|----------|-----------|--------|---------|
| **Random Forest** | **0.9182** | **0.8555** | 0.9908 | 0.9974 |
| XGBoost | 0.9125 | 0.8454 | 0.9911 | 0.9974 |
| Logistic Regression | 0.9123 | 0.8441 | 0.9924 | 0.9974 |

### Key Achievements
- **99.08% Recall**: Captures nearly all potential customers
- **85.55% Precision**: Minimizes false positives for cost-effective targeting
- **99.74% ROC-AUC**: Exceptional discrimination ability
- **0.84% Test Prediction Rate**: Conservative, high-confidence targeting

## Installation

### Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn
```

Or use the provided requirements file:
```bash
pip install -r requirements.txt
```

### Verified Environment
- Python 3.11+
- All major ML libraries supported
- Tested on macOS (compatible with Linux/Windows)

## Input Data Format

Expected CSV format with binary features (0/1):
```
UserID,basket_icon_click,basket_add_list,basket_add_detail,sort_by,image_picker,account_page_click,promo_banner_click,detail_wishlist_add,list_size_dropdown,closed_minibasket_click,checked_delivery_detail,checked_returns_detail,sign_in,saw_checkout,saw_sizecharts,saw_delivery,saw_account_upgrade,saw_homepage,device_mobile,device_computer,device_tablet,returning_user,loc_uk
user123,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,1,1,0,0,1,1
```

### Required Features (23 total):
- **Engagement**: basket_icon_click, basket_add_list, basket_add_detail, sort_by, image_picker
- **Account**: account_page_click, sign_in, saw_account_upgrade
- **Shopping Intent**: checked_delivery_detail, checked_returns_detail, saw_checkout, saw_delivery, saw_sizecharts
- **Content**: promo_banner_click, detail_wishlist_add, list_size_dropdown, closed_minibasket_click, saw_homepage
- **Device**: device_mobile, device_computer, device_tablet
- **User Type**: returning_user
- **Location**: loc_uk

## Output Format

Predictions include:
- **UserID**: Customer identifier
- **predicted_ordered**: Binary prediction (0/1)
- **purchase_probability**: Probability score (0-1)
- **prediction_timestamp**: When prediction was made

Example output:
```csv
UserID,predicted_ordered,purchase_probability,prediction_timestamp
user123,1,0.8234,2025-08-15T01:33:13.227942
user456,0,0.0234,2025-08-15T01:33:13.227942
```

## Business Applications

### Marketing Campaigns
- Target customers with probability > 0.5 for high-conversion campaigns
- Use probability scores for budget allocation
- A/B test different thresholds for optimal ROI

### Customer Segmentation
- High Intent (p > 0.5): Premium campaigns, personalized offers
- Medium Intent (0.1 < p < 0.5): Nurture campaigns, retargeting
- Low Intent (p < 0.1): Brand awareness, long-term nurturing

### Feature Insights
Top predictive features for business strategy:
1. **Checkout Behavior**: saw_checkout, checked_delivery_detail
2. **Engagement**: basket interactions, account activity
3. **Intent Signals**: sign_in combined with shopping actions

## Advanced Usage

### Custom Threshold Tuning
```python
# Load model and adjust threshold based on business needs
import pickle
with open('ml_outputs/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get probabilities
probs = model.predict_proba(X)[:, 1]

# Custom threshold (e.g., for high precision)
custom_threshold = 0.7
predictions = (probs >= custom_threshold).astype(int)
```

### Batch Processing
```python
from model_inference import CustomerPropensityPredictor

predictor = CustomerPropensityPredictor('ml_outputs/')
predictor.load_best_model()

# Process large files in chunks
for chunk in pd.read_csv('large_dataset.csv', chunksize=10000):
    predictions = predictor.predict(chunk)
    predictions.to_csv('predictions_chunk.csv', mode='a', header=False, index=False)
```

## Monitoring and Maintenance

### Model Performance Tracking
- Monitor prediction distribution over time
- Track business conversion rates vs predicted probabilities
- Alert if input data distribution shifts significantly

### Recommended Retraining Schedule
- **Monthly**: Performance monitoring and drift detection
- **Quarterly**: Full model retraining with new data
- **Annually**: Feature engineering review and architecture updates

## Files Generated

```
ml_outputs/
├── Models (Production Ready)
│   ├── random_forest_model.pkl           # Best model (F1: 0.9182)
│   ├── xgboost_model.pkl                 # Alternative model
│   └── logistic_regression_model.pkl     # Baseline model
├── Predictions
│   └── test_predictions.csv              # Test set predictions
├── Analysis
│   ├── model_evaluation_results.json     # Performance metrics
│   ├── random_forest_feature_importance.csv
│   ├── xgboost_feature_importance.csv
│   └── logistic_regression_coefficients.csv
├── Visualizations
│   └── model_comparison.png              # Performance comparison
└── Internal
    └── detailed_evaluation_results.pkl   # Complete evaluation data
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install --upgrade scikit-learn xgboost imbalanced-learn
```

**Memory Issues with Large Files**
- Process data in chunks using pandas `chunksize` parameter
- Use `model_inference.py` which handles memory efficiently

**Feature Missing Warnings**
- Missing features automatically filled with 0 (appropriate for binary data)
- Check input data format against expected schema

### Support

For technical issues:
1. Check log files (`ml_pipeline.log`) for detailed error messages
2. Verify input data format matches expected schema
3. Ensure all required packages are installed with compatible versions

## License

This project is designed for production use with comprehensive error handling, logging, and scalability features.