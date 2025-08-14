#!/usr/bin/env python3
"""
Model Inference Script
======================

Simple script to load trained models and make predictions on new data.
This demonstrates how to use the trained models in production.

Usage:
    python model_inference.py --input_file path/to/new_data.csv --output_file predictions.csv
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create interaction features between high-impact variables.
    
    This is a copy of the transformer used in training to ensure compatibility.
    """
    
    def __init__(self, high_impact_features: List[str] = None):
        # Top features based on uplift analysis
        self.high_impact_features = high_impact_features or [
            'checked_delivery_detail',
            'saw_checkout', 
            'sign_in',
            'basket_icon_click',
            'basket_add_detail',
            'closed_minibasket_click',
            'basket_add_list',
            'account_page_click'
        ]
        
    def fit(self, X, y=None):
        """Fit the transformer (no-op for interaction features)."""
        # Validate that required features exist
        missing_features = [f for f in self.high_impact_features if f not in X.columns]
        if missing_features:
            logger.warning(f"Missing high-impact features: {missing_features}")
            self.high_impact_features = [f for f in self.high_impact_features if f in X.columns]
        
        return self
    
    def transform(self, X):
        """Create interaction features."""
        X_copy = X.copy()
        
        # Create interaction features between top predictors
        interactions = [
            ('sign_in', 'saw_checkout'),  # Strong correlation found in analysis
            ('basket_icon_click', 'basket_add_detail'),  # Related basket actions
            ('basket_add_list', 'basket_add_detail'),  # Different basket actions
            ('checked_delivery_detail', 'sign_in'),  # Intent + engagement
            ('account_page_click', 'sign_in'),  # Account related actions
        ]
        
        for feat1, feat2 in interactions:
            if feat1 in X.columns and feat2 in X.columns:
                interaction_name = f'{feat1}_x_{feat2}'
                X_copy[interaction_name] = X_copy[feat1] * X_copy[feat2]
        
        # Create engagement score (sum of key engagement indicators)
        engagement_features = [
            'basket_icon_click', 'basket_add_list', 'basket_add_detail',
            'account_page_click', 'detail_wishlist_add', 'sort_by'
        ]
        available_engagement = [f for f in engagement_features if f in X.columns]
        if available_engagement:
            X_copy['engagement_score'] = X_copy[available_engagement].sum(axis=1)
        
        # Create intent score (sum of checkout-related actions)
        intent_features = [
            'saw_checkout', 'checked_delivery_detail', 'checked_returns_detail',
            'sign_in', 'saw_delivery'
        ]
        available_intent = [f for f in intent_features if f in X.columns]
        if available_intent:
            X_copy['intent_score'] = X_copy[available_intent].sum(axis=1)
        
        # Device interaction with user behavior
        device_cols = ['device_mobile', 'device_computer', 'device_tablet']
        available_devices = [d for d in device_cols if d in X.columns]
        
        for device in available_devices:
            if device in X.columns and 'returning_user' in X.columns:
                X_copy[f'{device}_returning'] = X_copy[device] * X_copy['returning_user']
        
        return X_copy


class CustomerPropensityPredictor:
    """
    Production inference class for customer propensity prediction.
    """
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = None
        self.model_name = None
        
    def load_best_model(self):
        """Load the best performing model (Random Forest based on our results)."""
        try:
            model_path = os.path.join(self.model_dir, 'random_forest_model.pkl')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.model_name = 'random_forest'
            logger.info(f"Loaded model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input data for prediction."""
        
        # Expected features (excluding UserID and target)
        expected_features = [
            'basket_icon_click', 'basket_add_list', 'basket_add_detail', 'sort_by',
            'image_picker', 'account_page_click', 'promo_banner_click', 
            'detail_wishlist_add', 'list_size_dropdown', 'closed_minibasket_click',
            'checked_delivery_detail', 'checked_returns_detail', 'sign_in',
            'saw_checkout', 'saw_sizecharts', 'saw_delivery', 'saw_account_upgrade',
            'saw_homepage', 'device_mobile', 'device_computer', 'device_tablet',
            'returning_user', 'loc_uk'
        ]
        
        # Check for required features
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features will be filled with 0: {missing_features}")
            
            # Fill missing features with 0 (appropriate for binary features)
            for feature in missing_features:
                df[feature] = 0
        
        # Select only the expected features
        feature_df = df[expected_features].copy()
        
        # Ensure all values are binary (0 or 1)
        for col in feature_df.columns:
            if feature_df[col].dtype not in ['int64', 'float64']:
                logger.warning(f"Converting {col} to numeric")
                feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce').fillna(0)
            
            # Clip values to 0-1 range
            feature_df[col] = feature_df[col].clip(0, 1)
        
        logger.info(f"Data validated. Shape: {feature_df.shape}")
        return feature_df
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on input data."""
        
        if self.model is None:
            raise ValueError("Model not loaded. Call load_best_model() first.")
        
        try:
            # Validate input data
            feature_df = self.validate_input_data(df)
            
            # Make predictions
            predictions = self.model.predict(feature_df)
            probabilities = self.model.predict_proba(feature_df)[:, 1]
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'predicted_ordered': predictions,
                'purchase_probability': probabilities
            })
            
            # Add UserID if available
            if 'UserID' in df.columns:
                results_df.insert(0, 'UserID', df['UserID'])
            else:
                results_df.insert(0, 'UserID', range(len(predictions)))
            
            # Add prediction timestamp
            results_df['prediction_timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Generated predictions for {len(results_df)} samples")
            logger.info(f"Predicted positive rate: {predictions.mean():.4f}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_from_file(self, input_path: str, output_path: str = None):
        """Load data from file, make predictions, and save results."""
        
        try:
            # Load input data
            logger.info(f"Loading data from: {input_path}")
            input_df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(input_df)} samples")
            
            # Make predictions
            results_df = self.predict(input_df)
            
            # Save results
            if output_path is None:
                output_path = input_path.replace('.csv', '_predictions.csv')
            
            results_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to: {output_path}")
            
            # Print summary
            positive_predictions = (results_df['predicted_ordered'] == 1).sum()
            logger.info(f"Summary:")
            logger.info(f"  Total predictions: {len(results_df)}")
            logger.info(f"  Predicted to purchase: {positive_predictions}")
            logger.info(f"  Predicted purchase rate: {positive_predictions/len(results_df):.4f}")
            logger.info(f"  Average purchase probability: {results_df['purchase_probability'].mean():.4f}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise


def main():
    """Main function for command-line usage."""
    
    parser = argparse.ArgumentParser(description='Customer Propensity Prediction Inference')
    parser.add_argument('--input_file', required=True, 
                       help='Path to input CSV file with customer data')
    parser.add_argument('--output_file', 
                       help='Path to output CSV file for predictions (optional)')
    parser.add_argument('--model_dir', 
                       default='/Users/jerrylaivivemachi/DS PROJECT/J_DA_Project/Customer propensity to purchase dataset/ml_outputs',
                       help='Directory containing trained models')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    if not os.path.exists(args.model_dir):
        logger.error(f"Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    try:
        # Initialize predictor
        predictor = CustomerPropensityPredictor(args.model_dir)
        predictor.load_best_model()
        
        # Make predictions
        results = predictor.predict_from_file(args.input_file, args.output_file)
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()