#!/usr/bin/env python3
"""
Customer Propensity to Purchase - ML Pipeline MVP
==================================================

A comprehensive machine learning pipeline for predicting customer purchase propensity
with robust feature engineering, class imbalance handling, and multiple model evaluation.

Author: Senior ML Engineer
Date: 2025-08-14
"""

import os
import sys
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Handle optional dependencies
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("imbalanced-learn not available. Install with: pip install imbalanced-learn")

import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class InteractionFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to create interaction features between high-impact variables.
    
    Based on the analysis results, we'll create interactions between the most
    predictive features to capture non-linear relationships.
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
        
        logger.info(f"Created {len(X_copy.columns) - len(X.columns)} interaction features")
        return X_copy


class CustomerPropensityPipeline:
    """
    Comprehensive ML Pipeline for Customer Propensity Analysis
    
    Handles data loading, feature engineering, model training, evaluation,
    and prediction generation with proper error handling and logging.
    """
    
    def __init__(self, 
                 train_path: str,
                 test_path: str,
                 output_dir: str = "ml_outputs",
                 random_state: int = 42):
        
        self.train_path = train_path
        self.test_path = test_path
        self.output_dir = output_dir
        self.random_state = random_state
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize containers
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.models = {}
        self.feature_names = []
        self.results = {}
        
        logger.info(f"Pipeline initialized with output directory: {output_dir}")
    
    def load_data(self) -> None:
        """Load training and testing data with error handling."""
        try:
            logger.info("Loading training and testing data...")
            
            self.train_data = pd.read_csv(self.train_path)
            self.test_data = pd.read_csv(self.test_path)
            
            logger.info(f"Training data shape: {self.train_data.shape}")
            logger.info(f"Testing data shape: {self.test_data.shape}")
            
            # Basic validation
            if 'ordered' not in self.train_data.columns:
                raise ValueError("Target column 'ordered' not found in training data")
            
            # Check for missing values
            train_missing = self.train_data.isnull().sum().sum()
            test_missing = self.test_data.isnull().sum().sum()
            
            if train_missing > 0:
                logger.warning(f"Training data has {train_missing} missing values")
            if test_missing > 0:
                logger.warning(f"Testing data has {test_missing} missing values")
                
            # Log class distribution
            class_dist = self.train_data['ordered'].value_counts(normalize=True)
            logger.info(f"Class distribution: {dict(class_dist)}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features(self) -> None:
        """Prepare features and target variables with train/validation split."""
        try:
            logger.info("Preparing features and splitting data...")
            
            # Separate features and target
            feature_cols = [col for col in self.train_data.columns 
                          if col not in ['UserID', 'ordered']]
            
            X = self.train_data[feature_cols]
            y = self.train_data['ordered']
            
            # Stratified train/validation split
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X, y, 
                test_size=0.2,
                random_state=self.random_state,
                stratify=y
            )
            
            # Store feature names for later use
            self.feature_names = feature_cols.copy()
            
            logger.info(f"Training set size: {len(self.X_train)}")
            logger.info(f"Validation set size: {len(self.X_val)}")
            logger.info(f"Number of features: {len(feature_cols)}")
            
            # Log class distribution in splits
            train_dist = self.y_train.value_counts(normalize=True)
            val_dist = self.y_val.value_counts(normalize=True)
            
            logger.info(f"Training class distribution: {dict(train_dist)}")
            logger.info(f"Validation class distribution: {dict(val_dist)}")
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def build_preprocessing_pipeline(self, use_smote: bool = True) -> Pipeline:
        """Build preprocessing pipeline with feature engineering and optional SMOTE."""
        
        # Feature engineering transformer
        interaction_transformer = InteractionFeatureTransformer()
        
        # Standard scaling for numerical features (though all are binary in this case)
        # We'll keep it for the interaction features we create
        scaler = StandardScaler()
        
        if IMBLEARN_AVAILABLE and use_smote:
            # Use SMOTE for handling class imbalance
            smote = SMOTE(random_state=self.random_state, k_neighbors=5)
            
            preprocessing_pipeline = ImbPipeline([
                ('interactions', interaction_transformer),
                ('scaler', scaler),
                ('smote', smote)
            ])
        else:
            preprocessing_pipeline = Pipeline([
                ('interactions', interaction_transformer),
                ('scaler', scaler)
            ])
            
            if use_smote:
                logger.warning("SMOTE requested but imbalanced-learn not available. "
                             "Using class weights instead.")
        
        return preprocessing_pipeline
    
    def train_models(self) -> None:
        """Train multiple models with different approaches to class imbalance."""
        logger.info("Starting model training...")
        
        models_config = {
            'logistic_regression': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    class_weight='balanced',
                    max_iter=1000
                ),
                'use_smote': False,
                'param_grid': {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__l1_ratio': [0.1, 0.5, 0.9]
                }
            },
            'logistic_regression_smote': {
                'model': LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000
                ),
                'use_smote': True,
                'param_grid': {
                    'classifier__C': [0.1, 1.0, 10.0]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=-1
                ),
                'use_smote': False,
                'param_grid': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__min_samples_split': [2, 5]
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            # Calculate scale_pos_weight for class imbalance
            scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
            
            models_config['xgboost'] = {
                'model': xgb.XGBClassifier(
                    random_state=self.random_state,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss'
                ),
                'use_smote': False,
                'param_grid': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__max_depth': [6, 10],
                    'classifier__learning_rate': [0.1, 0.01]
                }
            }
        
        # Train each model
        for model_name, config in models_config.items():
            try:
                logger.info(f"Training {model_name}...")
                
                # Skip SMOTE models if imblearn not available
                if config['use_smote'] and not IMBLEARN_AVAILABLE:
                    logger.warning(f"Skipping {model_name} - requires imbalanced-learn")
                    continue
                
                # Build pipeline
                preprocessing = self.build_preprocessing_pipeline(config['use_smote'])
                
                if IMBLEARN_AVAILABLE and config['use_smote']:
                    pipeline = ImbPipeline([
                        ('preprocessor', preprocessing),
                        ('classifier', config['model'])
                    ])
                else:
                    pipeline = Pipeline([
                        ('preprocessor', preprocessing),
                        ('classifier', config['model'])
                    ])
                
                # Grid search with cross-validation
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
                
                grid_search = GridSearchCV(
                    pipeline,
                    config['param_grid'],
                    cv=cv,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0
                )
                
                # Fit the model
                grid_search.fit(self.X_train, self.y_train)
                
                # Store the best model
                self.models[model_name] = grid_search.best_estimator_
                
                logger.info(f"{model_name} training completed")
                logger.info(f"Best parameters: {grid_search.best_params_}")
                logger.info(f"Best CV F1 score: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
    
    def evaluate_models(self) -> None:
        """Evaluate all trained models on validation set."""
        logger.info("Evaluating models on validation set...")
        
        self.results = {}
        
        for model_name, model in self.models.items():
            try:
                logger.info(f"Evaluating {model_name}...")
                
                # Make predictions
                y_pred = model.predict(self.X_val)
                y_pred_proba = model.predict_proba(self.X_val)[:, 1]
                
                # Calculate metrics
                metrics = {
                    'f1': f1_score(self.y_val, y_pred),
                    'precision': precision_score(self.y_val, y_pred),
                    'recall': recall_score(self.y_val, y_pred),
                    'roc_auc': roc_auc_score(self.y_val, y_pred_proba),
                    'avg_precision': average_precision_score(self.y_val, y_pred_proba)
                }
                
                # Store results
                self.results[model_name] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'classification_report': classification_report(self.y_val, y_pred, output_dict=True)
                }
                
                logger.info(f"{model_name} - F1: {metrics['f1']:.4f}, "
                           f"Precision: {metrics['precision']:.4f}, "
                           f"Recall: {metrics['recall']:.4f}, "
                           f"ROC-AUC: {metrics['roc_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
    
    def analyze_feature_importance(self) -> None:
        """Analyze feature importance for tree-based models."""
        logger.info("Analyzing feature importance...")
        
        for model_name, model in self.models.items():
            try:
                # Get the final classifier from pipeline
                classifier = model.named_steps['classifier']
                
                if hasattr(classifier, 'feature_importances_'):
                    # Tree-based models
                    
                    # Get feature names after preprocessing
                    # Apply preprocessing to get the final feature names
                    X_transformed = model.named_steps['preprocessor'].transform(self.X_train[:100])
                    
                    if hasattr(X_transformed, 'toarray'):
                        X_transformed = X_transformed.toarray()
                    
                    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
                    
                    # If we can get better names, use those
                    if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
                        try:
                            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                        except:
                            pass
                    
                    importances = classifier.feature_importances_
                    
                    # Create feature importance DataFrame
                    feature_importance_df = pd.DataFrame({
                        'feature': feature_names[:len(importances)],
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    # Save to CSV
                    output_path = os.path.join(self.output_dir, f'{model_name}_feature_importance.csv')
                    feature_importance_df.to_csv(output_path, index=False)
                    
                    logger.info(f"Feature importance saved for {model_name}")
                    logger.info(f"Top 5 features: {list(feature_importance_df.head()['feature'])}")
                
                elif hasattr(classifier, 'coef_'):
                    # Linear models
                    X_transformed = model.named_steps['preprocessor'].transform(self.X_train[:100])
                    if hasattr(X_transformed, 'toarray'):
                        X_transformed = X_transformed.toarray()
                    
                    feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
                    
                    coefficients = classifier.coef_[0] if classifier.coef_.ndim > 1 else classifier.coef_
                    
                    # Create coefficient DataFrame
                    coef_df = pd.DataFrame({
                        'feature': feature_names[:len(coefficients)],
                        'coefficient': coefficients,
                        'abs_coefficient': np.abs(coefficients)
                    }).sort_values('abs_coefficient', ascending=False)
                    
                    # Save to CSV
                    output_path = os.path.join(self.output_dir, f'{model_name}_coefficients.csv')
                    coef_df.to_csv(output_path, index=False)
                    
                    logger.info(f"Coefficients saved for {model_name}")
                
            except Exception as e:
                logger.error(f"Error analyzing feature importance for {model_name}: {str(e)}")
                continue
    
    def save_models_and_results(self) -> None:
        """Save trained models and evaluation results."""
        logger.info("Saving models and results...")
        
        try:
            # Save models
            for model_name, model in self.models.items():
                model_path = os.path.join(self.output_dir, f'{model_name}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"Model saved: {model_path}")
            
            # Save results summary
            results_summary = {}
            for model_name, result in self.results.items():
                results_summary[model_name] = result['metrics']
            
            results_path = os.path.join(self.output_dir, 'model_evaluation_results.json')
            with open(results_path, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            # Save detailed results
            detailed_results_path = os.path.join(self.output_dir, 'detailed_evaluation_results.pkl')
            with open(detailed_results_path, 'wb') as f:
                pickle.dump(self.results, f)
            
            logger.info("Results saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models and results: {str(e)}")
            raise
    
    def generate_test_predictions(self) -> None:
        """Generate predictions on test set using the best model."""
        logger.info("Generating predictions on test set...")
        
        try:
            # Find best model based on F1 score
            best_model_name = max(self.results.keys(), 
                                key=lambda x: self.results[x]['metrics']['f1'])
            
            best_model = self.models[best_model_name]
            
            logger.info(f"Using best model: {best_model_name}")
            logger.info(f"Best F1 score: {self.results[best_model_name]['metrics']['f1']:.4f}")
            
            # Prepare test features
            test_feature_cols = [col for col in self.test_data.columns 
                               if col not in ['UserID']]
            
            # Handle missing target column in test set
            if 'ordered' in test_feature_cols:
                test_feature_cols.remove('ordered')
            
            X_test = self.test_data[test_feature_cols]
            
            # Generate predictions
            test_predictions = best_model.predict(X_test)
            test_probabilities = best_model.predict_proba(X_test)[:, 1]
            
            # Create predictions DataFrame
            predictions_df = pd.DataFrame({
                'UserID': self.test_data['UserID'] if 'UserID' in self.test_data.columns 
                         else range(len(test_predictions)),
                'predicted_ordered': test_predictions,
                'purchase_probability': test_probabilities
            })
            
            # Save predictions
            pred_path = os.path.join(self.output_dir, 'test_predictions.csv')
            predictions_df.to_csv(pred_path, index=False)
            
            logger.info(f"Test predictions saved: {pred_path}")
            logger.info(f"Predicted positive rate: {test_predictions.mean():.4f}")
            
        except Exception as e:
            logger.error(f"Error generating test predictions: {str(e)}")
            raise
    
    def create_evaluation_plots(self) -> None:
        """Create evaluation plots for model comparison."""
        logger.info("Creating evaluation plots...")
        
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Model comparison metrics plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            metrics_to_plot = ['f1', 'precision', 'recall', 'roc_auc']
            
            for idx, metric in enumerate(metrics_to_plot):
                ax = axes[idx // 2, idx % 2]
                
                model_names = list(self.results.keys())
                metric_values = [self.results[model][metric] for model in model_names]
                
                bars = ax.bar(model_names, metric_values)
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'model_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # ROC curves
            plt.figure(figsize=(10, 8))
            
            for model_name in self.results.keys():
                y_pred_proba = self.results[model_name]['probabilities']
                fpr, tpr, _ = roc_curve(self.y_val, y_pred_proba)
                auc_score = self.results[model_name]['metrics']['roc_auc']
                
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            roc_path = os.path.join(self.output_dir, 'roc_curves.png')
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Precision-Recall curves
            plt.figure(figsize=(10, 8))
            
            for model_name in self.results.keys():
                y_pred_proba = self.results[model_name]['probabilities']
                precision, recall, _ = precision_recall_curve(self.y_val, y_pred_proba)
                avg_precision = self.results[model_name]['metrics']['avg_precision']
                
                plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            pr_path = os.path.join(self.output_dir, 'precision_recall_curves.png')
            plt.savefig(pr_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Evaluation plots created successfully")
            
        except Exception as e:
            logger.error(f"Error creating plots: {str(e)}")
    
    def run_pipeline(self) -> None:
        """Run the complete ML pipeline."""
        try:
            logger.info("=" * 60)
            logger.info("CUSTOMER PROPENSITY ML PIPELINE STARTED")
            logger.info("=" * 60)
            
            # Pipeline steps
            self.load_data()
            self.prepare_features()
            self.train_models()
            self.evaluate_models()
            self.analyze_feature_importance()
            self.save_models_and_results()
            self.generate_test_predictions()
            self.create_evaluation_plots()
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            # Print summary
            logger.info("\nMODEL PERFORMANCE SUMMARY:")
            logger.info("-" * 40)
            
            for model_name, result in self.results.items():
                metrics = result['metrics']
                logger.info(f"{model_name}:")
                logger.info(f"  F1: {metrics['f1']:.4f}")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall: {metrics['recall']:.4f}")
                logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
                logger.info("-" * 40)
            
            # Best model
            if self.results:
                best_model_name = max(self.results.keys(), 
                                    key=lambda x: self.results[x]['metrics']['f1'])
                
                logger.info(f"\nBEST MODEL: {best_model_name}")
                logger.info(f"F1 Score: {self.results[best_model_name]['metrics']['f1']:.4f}")
            
            logger.info(f"\nAll outputs saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the ML pipeline."""
    
    # Configuration
    BASE_DIR = "/Users/jerrylaivivemachi/DS PROJECT/J_DA_Project/Customer propensity to purchase dataset"
    TRAIN_PATH = os.path.join(BASE_DIR, "training_sample.csv")
    TEST_PATH = os.path.join(BASE_DIR, "testing_sample.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "ml_outputs")
    
    # Validate input files
    if not os.path.exists(TRAIN_PATH):
        logger.error(f"Training file not found: {TRAIN_PATH}")
        sys.exit(1)
    
    if not os.path.exists(TEST_PATH):
        logger.error(f"Testing file not found: {TEST_PATH}")
        sys.exit(1)
    
    # Initialize and run pipeline
    pipeline = CustomerPropensityPipeline(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        output_dir=OUTPUT_DIR,
        random_state=42
    )
    
    # Run the complete pipeline
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()