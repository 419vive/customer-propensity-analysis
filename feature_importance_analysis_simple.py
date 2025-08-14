#!/usr/bin/env python3
"""
Feature Importance Analysis and Business Insights
================================================

Comprehensive analysis of feature importance using existing model outputs:
- Business interpretation of key features
- Feature interaction analysis
- Actionable insights for marketing teams

Author: Senior ML Engineer
Date: 2025-08-14
"""

import os
import sys
import logging
import warnings
import json
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class FeatureImportanceAnalyzer:
    """
    Feature importance analysis using existing model outputs.
    """
    
    def __init__(self, 
                 models_dir: str = "ml_outputs",
                 output_dir: str = "feature_analysis_outputs"):
        
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.feature_importance_data = {}
        self.business_insights = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Feature name mapping (from feature_X to actual names)
        self.feature_mapping = {
            'feature_0': 'basket_icon_click',
            'feature_1': 'basket_add_list', 
            'feature_2': 'basket_add_detail',
            'feature_3': 'sort_by',
            'feature_4': 'image_picker',
            'feature_5': 'account_page_click',
            'feature_6': 'promo_banner_click',
            'feature_7': 'detail_wishlist_add',
            'feature_8': 'list_size_dropdown',
            'feature_9': 'closed_minibasket_click',
            'feature_10': 'checked_delivery_detail',
            'feature_11': 'checked_returns_detail',
            'feature_12': 'sign_in',
            'feature_13': 'saw_checkout',
            'feature_14': 'saw_sizecharts',
            'feature_15': 'saw_delivery',
            'feature_16': 'saw_account_upgrade',
            'feature_17': 'saw_homepage',
            'feature_18': 'device_mobile',
            'feature_19': 'device_computer',
            'feature_20': 'device_tablet',
            'feature_21': 'returning_user',
            'feature_22': 'loc_uk',
            'feature_23': 'sign_in_x_saw_checkout',
            'feature_24': 'basket_icon_click_x_basket_add_detail',
            'feature_25': 'basket_add_list_x_basket_add_detail',
            'feature_26': 'checked_delivery_detail_x_sign_in',
            'feature_27': 'account_page_click_x_sign_in',
            'feature_28': 'engagement_score',
            'feature_29': 'intent_score',
            'feature_30': 'device_mobile_returning',
            'feature_31': 'device_computer_returning',
            'feature_32': 'device_tablet_returning'
        }
        
        logger.info("Feature importance analyzer initialized")
    
    def load_existing_importance_data(self) -> None:
        """Load existing feature importance data from CSV files."""
        logger.info("Loading existing feature importance data...")
        
        try:
            # Load Random Forest feature importance
            rf_path = os.path.join(self.models_dir, 'random_forest_feature_importance.csv')
            if os.path.exists(rf_path):
                rf_importance = pd.read_csv(rf_path)
                rf_importance['feature_name'] = rf_importance['feature'].map(self.feature_mapping).fillna(rf_importance['feature'])
                rf_importance['model'] = 'Random Forest'
                self.feature_importance_data['random_forest'] = rf_importance
                logger.info("Loaded Random Forest feature importance")
            
            # Load XGBoost feature importance
            xgb_path = os.path.join(self.models_dir, 'xgboost_feature_importance.csv')
            if os.path.exists(xgb_path):
                xgb_importance = pd.read_csv(xgb_path)
                xgb_importance['feature_name'] = xgb_importance['feature'].map(self.feature_mapping).fillna(xgb_importance['feature'])
                xgb_importance['model'] = 'XGBoost'
                self.feature_importance_data['xgboost'] = xgb_importance
                logger.info("Loaded XGBoost feature importance")
            
            # Load Logistic Regression coefficients
            lr_path = os.path.join(self.models_dir, 'logistic_regression_coefficients.csv')
            if os.path.exists(lr_path):
                lr_coefs = pd.read_csv(lr_path)
                lr_coefs['feature_name'] = lr_coefs['feature'].map(self.feature_mapping).fillna(lr_coefs['feature'])
                lr_coefs['model'] = 'Logistic Regression'
                lr_coefs['importance'] = lr_coefs['abs_coefficient']  # Use absolute coefficient as importance
                self.feature_importance_data['logistic_regression'] = lr_coefs
                logger.info("Loaded Logistic Regression coefficients")
            
        except Exception as e:
            logger.error(f"Error loading importance data: {str(e)}")
            raise
    
    def analyze_top_features(self) -> Dict[str, Any]:
        """Analyze top features across all models."""
        logger.info("Analyzing top features...")
        
        try:
            # Combine importance scores across models
            all_features = {}
            
            for model_name, data in self.feature_importance_data.items():
                for _, row in data.iterrows():
                    feature_name = row['feature_name']
                    importance = row['importance']
                    
                    if feature_name not in all_features:
                        all_features[feature_name] = []
                    all_features[feature_name].append(importance)
            
            # Calculate average importance
            feature_summary = []
            for feature, importances in all_features.items():
                feature_summary.append({
                    'feature': feature,
                    'avg_importance': np.mean(importances),
                    'max_importance': np.max(importances),
                    'min_importance': np.min(importances),
                    'std_importance': np.std(importances),
                    'models_count': len(importances),
                    'business_category': self._categorize_feature(feature),
                    'business_meaning': self._get_business_meaning(feature)
                })
            
            # Sort by average importance
            feature_summary = sorted(feature_summary, key=lambda x: x['avg_importance'], reverse=True)
            
            # Convert to DataFrame
            summary_df = pd.DataFrame(feature_summary)
            
            # Save to CSV
            output_path = os.path.join(self.output_dir, 'consolidated_feature_importance.csv')
            summary_df.to_csv(output_path, index=False)
            
            logger.info("Top features analysis completed")
            return {'summary': feature_summary, 'dataframe': summary_df}
            
        except Exception as e:
            logger.error(f"Error analyzing top features: {str(e)}")
            return {}
    
    def _categorize_feature(self, feature: str) -> str:
        """Categorize feature by business function."""
        feature_lower = feature.lower()
        
        if any(word in feature_lower for word in ['basket', 'cart', 'add', 'wishlist']):
            return 'Shopping Behavior'
        elif any(word in feature_lower for word in ['checkout', 'delivery', 'returns', 'sign_in']):
            return 'Purchase Intent'
        elif any(word in feature_lower for word in ['device', 'mobile', 'computer', 'tablet']):
            return 'Device & Context'
        elif any(word in feature_lower for word in ['engagement', 'intent', 'score', '_x_']):
            return 'Derived Metrics'
        elif any(word in feature_lower for word in ['saw', 'homepage', 'account', 'promo']):
            return 'Site Navigation'
        else:
            return 'Other'
    
    def _get_business_meaning(self, feature: str) -> str:
        """Get business interpretation of feature."""
        meanings = {
            'checked_delivery_detail': 'Customer actively seeking delivery information - strong purchase intent signal',
            'saw_checkout': 'Customer reached checkout page - very high conversion indicator',
            'sign_in': 'Customer authentication - indicates committed, returning user behavior',
            'basket_icon_click': 'Customer engaging with shopping cart - moderate to high intent',
            'basket_add_detail': 'Customer adding items from product detail pages - strong intent signal',
            'basket_add_list': 'Customer adding items from category/list pages - moderate intent',
            'closed_minibasket_click': 'Customer reviewing mini cart contents - purchase consideration phase',
            'account_page_click': 'Customer accessing account area - typically returning user behavior',
            'checked_returns_detail': 'Customer researching return policy - risk mitigation behavior',
            'saw_delivery': 'Customer viewing delivery options - purchase planning signal',
            'detail_wishlist_add': 'Customer saving items for future - future purchase pipeline',
            'device_mobile': 'Mobile device usage - affects UX and conversion patterns',
            'device_computer': 'Desktop usage - typically associated with higher conversion rates',
            'device_tablet': 'Tablet usage - hybrid browsing and purchasing behavior',
            'returning_user': 'Customer familiarity and loyalty indicator',
            'engagement_score': 'Composite metric of site engagement activities',
            'intent_score': 'Composite metric of purchase intention signals',
            'checked_delivery_detail_x_sign_in': 'High-value interaction: authenticated user checking delivery',
            'sign_in_x_saw_checkout': 'Critical interaction: authenticated user at checkout',
            'basket_icon_click_x_basket_add_detail': 'Shopping flow optimization opportunity',
            'loc_uk': 'Geographic market indicator affecting shipping and pricing'
        }
        
        if feature in meanings:
            return meanings[feature]
        
        # Generate meaning for interaction features
        if '_x_' in feature:
            return f"Interaction effect capturing combined impact of {feature.replace('_x_', ' and ')}"
        elif 'score' in feature.lower():
            return f"Derived behavioral metric: {feature.replace('_', ' ').title()}"
        else:
            return f"User behavior signal: {feature.replace('_', ' ').title()}"
    
    def generate_business_insights(self, feature_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable business insights."""
        logger.info("Generating business insights...")
        
        insights = {
            'key_findings': [],
            'category_insights': {},
            'recommendations': [],
            'strategic_priorities': []
        }
        
        try:
            if 'summary' not in feature_analysis:
                return insights
            
            top_features = feature_analysis['summary'][:15]  # Top 15 features
            
            # Key findings
            insights['key_findings'] = [
                f"Top conversion driver: {top_features[0]['feature']} (Importance: {top_features[0]['avg_importance']:.3f})",
                f"Most consistent predictor across models: {max(top_features, key=lambda x: 1/x['std_importance'] if x['std_importance'] > 0 else 0)['feature']}",
                f"Strongest derived metric: {max([f for f in top_features if f['business_category'] == 'Derived Metrics'], key=lambda x: x['avg_importance'], default={'feature': 'None', 'avg_importance': 0})['feature']}"
            ]
            
            # Category insights
            categories = {}
            for feature in top_features:
                cat = feature['business_category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(feature)
            
            for category, features in categories.items():
                avg_importance = np.mean([f['avg_importance'] for f in features])
                top_feature = max(features, key=lambda x: x['avg_importance'])
                
                insights['category_insights'][category] = {
                    'average_importance': avg_importance,
                    'feature_count': len(features),
                    'top_feature': top_feature['feature'],
                    'business_impact': self._get_category_impact(category)
                }
            
            # Recommendations
            insights['recommendations'] = self._generate_recommendations(top_features, categories)
            
            # Strategic priorities
            insights['strategic_priorities'] = self._generate_strategic_priorities(top_features, categories)
            
            # Save insights
            output_path = os.path.join(self.output_dir, 'business_insights.json')
            with open(output_path, 'w') as f:
                json.dump(insights, f, indent=2, default=str)
            
            logger.info("Business insights generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating business insights: {str(e)}")
        
        return insights
    
    def _get_category_impact(self, category: str) -> str:
        """Get business impact description for category."""
        impacts = {
            'Shopping Behavior': 'Direct impact on conversion funnel - optimize cart and wishlist experiences',
            'Purchase Intent': 'Critical conversion signals - focus on reducing friction at these touchpoints',
            'Device & Context': 'Platform optimization opportunities - ensure consistent experience across devices',
            'Derived Metrics': 'Composite signals for advanced targeting - use for segmentation and personalization',
            'Site Navigation': 'User journey optimization - improve navigation and content discovery',
            'Other': 'Additional factors requiring investigation and optimization'
        }
        return impacts.get(category, 'Requires further analysis for business impact assessment')
    
    def _generate_recommendations(self, top_features: List[Dict], categories: Dict[str, List]) -> List[str]:
        """Generate specific actionable recommendations."""
        recommendations = []
        
        # Top feature specific recommendations
        top_5_features = [f['feature'] for f in top_features[:5]]
        
        for feature in top_5_features:
            if 'checkout' in feature.lower():
                recommendations.append(f"Optimize checkout experience: {feature} is a top predictor. Implement one-page checkout, guest checkout options, and clear progress indicators.")
            elif 'delivery' in feature.lower():
                recommendations.append(f"Enhance delivery communication: {feature} strongly predicts conversion. Display delivery options prominently and offer multiple delivery speeds.")
            elif 'sign_in' in feature.lower():
                recommendations.append(f"Streamline authentication: {feature} is crucial for conversion. Implement social login, remember user preferences, and reduce sign-in friction.")
            elif 'basket' in feature.lower():
                recommendations.append(f"Optimize cart experience: {feature} influences purchase decisions. Implement persistent cart, cart recommendations, and clear pricing.")
            elif '_x_' in feature:
                recommendations.append(f"Focus on user journey optimization: {feature} shows important behavioral interactions. Create seamless flows between these actions.")
        
        # Category-based recommendations
        if 'Shopping Behavior' in categories:
            recommendations.append("Shopping Behavior Priority: Implement advanced cart features like saved items, price alerts, and smart recommendations.")
        
        if 'Purchase Intent' in categories:
            recommendations.append("Purchase Intent Priority: Reduce checkout abandonment with exit-intent popups, cart recovery emails, and simplified payment options.")
        
        if 'Device & Context' in categories:
            recommendations.append("Device Optimization: Ensure mobile-first design, touch-friendly interfaces, and consistent functionality across all devices.")
        
        return recommendations[:10]  # Top 10 recommendations
    
    def _generate_strategic_priorities(self, top_features: List[Dict], categories: Dict[str, List]) -> List[str]:
        """Generate strategic priorities for business leadership."""
        priorities = [
            "Customer Journey Optimization: Focus on the critical touchpoints identified by the top predictive features",
            "Personalization Engine: Leverage behavioral signals for targeted marketing and product recommendations",
            "Conversion Rate Optimization: Systematic testing of the highest-impact features and user flows",
            "Mobile Experience: Prioritize mobile optimization given device-related feature importance",
            "Authentication Strategy: Simplify and incentivize user registration and sign-in processes"
        ]
        
        # Add category-specific priorities
        category_importance = {cat: np.mean([f['avg_importance'] for f in features]) 
                             for cat, features in categories.items()}
        
        top_category = max(category_importance, key=category_importance.get)
        priorities.insert(0, f"{top_category} Focus: This category shows the highest predictive power - allocate primary optimization resources here")
        
        return priorities
    
    def create_visualizations(self, feature_analysis: Dict[str, Any]) -> None:
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations...")
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            if 'dataframe' not in feature_analysis:
                return
            
            df = feature_analysis['dataframe']
            
            # 1. Top Features Comparison
            fig, axes = plt.subplots(2, 2, figsize=(18, 14))
            
            # Feature importance by model
            ax1 = axes[0, 0]
            top_15 = df.head(15)
            
            # Plot for each model
            model_colors = {'Random Forest': 'blue', 'XGBoost': 'orange', 'Logistic Regression': 'green'}
            
            for i, (model_name, model_data) in enumerate(self.feature_importance_data.items()):
                model_features = model_data.set_index('feature_name')['importance']
                model_top_15 = model_features.reindex(top_15['feature']).fillna(0)
                
                x_pos = np.arange(len(top_15)) + i * 0.25
                ax1.bar(x_pos, model_top_15.values, width=0.25, 
                       label=model_data['model'].iloc[0], alpha=0.8)
            
            ax1.set_xlabel('Features')
            ax1.set_ylabel('Importance')
            ax1.set_title('Feature Importance Comparison Across Models')
            ax1.set_xticks(np.arange(len(top_15)) + 0.25)
            ax1.set_xticklabels(top_15['feature'], rotation=45, ha='right')
            ax1.legend()
            
            # Category distribution
            ax2 = axes[0, 1]
            category_counts = df['business_category'].value_counts()
            ax2.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
            ax2.set_title('Feature Distribution by Business Category')
            
            # Category importance
            ax3 = axes[1, 0]
            category_importance = df.groupby('business_category')['avg_importance'].mean().sort_values(ascending=True)
            ax3.barh(range(len(category_importance)), category_importance.values)
            ax3.set_yticks(range(len(category_importance)))
            ax3.set_yticklabels(category_importance.index)
            ax3.set_xlabel('Average Importance')
            ax3.set_title('Average Importance by Business Category')
            
            # Top 10 features with business meaning
            ax4 = axes[1, 1]
            top_10 = df.head(10)
            bars = ax4.barh(range(len(top_10)), top_10['avg_importance'])
            ax4.set_yticks(range(len(top_10)))
            ax4.set_yticklabels(top_10['feature'])
            ax4.set_xlabel('Average Importance')
            ax4.set_title('Top 10 Most Important Features')
            ax4.invert_yaxis()
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax4.text(width, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'comprehensive_feature_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Feature importance heatmap
            self._create_feature_heatmap(df)
            
            logger.info("Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def _create_feature_heatmap(self, df: pd.DataFrame) -> None:
        """Create heatmap of feature importance by category and model."""
        try:
            # Prepare data for heatmap
            heatmap_data = []
            
            for _, row in df.head(20).iterrows():  # Top 20 features
                feature_name = row['feature']
                
                # Get importance for each model
                feature_row = {'Feature': feature_name, 'Category': row['business_category']}
                
                for model_name, model_data in self.feature_importance_data.items():
                    model_importance = model_data[model_data['feature_name'] == feature_name]
                    if not model_importance.empty:
                        feature_row[model_data['model'].iloc[0]] = model_importance['importance'].iloc[0]
                    else:
                        feature_row[model_data['model'].iloc[0]] = 0
                
                heatmap_data.append(feature_row)
            
            heatmap_df = pd.DataFrame(heatmap_data)
            heatmap_df = heatmap_df.set_index('Feature')
            
            # Remove non-numeric columns for heatmap
            numeric_cols = [col for col in heatmap_df.columns if col != 'Category']
            heatmap_numeric = heatmap_df[numeric_cols]
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(heatmap_numeric, annot=True, fmt='.3f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Feature Importance'})
            plt.title('Feature Importance Heatmap by Model')
            plt.xlabel('Model')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create feature heatmap: {str(e)}")
    
    def run_comprehensive_analysis(self) -> None:
        """Run the complete feature importance analysis."""
        try:
            logger.info("=" * 60)
            logger.info("FEATURE IMPORTANCE ANALYSIS STARTED")
            logger.info("=" * 60)
            
            # Load existing data
            self.load_existing_importance_data()
            
            # Analyze features
            feature_analysis = self.analyze_top_features()
            
            # Generate insights
            business_insights = self.generate_business_insights(feature_analysis)
            
            # Create visualizations
            self.create_visualizations(feature_analysis)
            
            # Store results
            self.business_insights = business_insights
            
            logger.info("=" * 60)
            logger.info("FEATURE IMPORTANCE ANALYSIS COMPLETED")
            logger.info("=" * 60)
            
            # Print summary
            self._print_analysis_summary(feature_analysis, business_insights)
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
    
    def _print_analysis_summary(self, feature_analysis: Dict[str, Any], business_insights: Dict[str, Any]) -> None:
        """Print summary of analysis results."""
        try:
            logger.info("\nFEATURE IMPORTANCE ANALYSIS SUMMARY:")
            logger.info("-" * 50)
            
            if 'summary' in feature_analysis:
                top_10_features = feature_analysis['summary'][:10]
                
                logger.info("\nTOP 10 MOST IMPORTANT FEATURES:")
                for i, feature in enumerate(top_10_features, 1):
                    logger.info(f"{i:2d}. {feature['feature']:<35} (Importance: {feature['avg_importance']:.4f})")
                
            if 'key_findings' in business_insights:
                logger.info("\nKEY BUSINESS FINDINGS:")
                for i, finding in enumerate(business_insights['key_findings'], 1):
                    logger.info(f"{i}. {finding}")
            
            if 'recommendations' in business_insights:
                logger.info("\nTOP BUSINESS RECOMMENDATIONS:")
                for i, rec in enumerate(business_insights['recommendations'][:5], 1):
                    logger.info(f"{i}. {rec}")
            
            logger.info(f"\nAll analysis outputs saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error printing summary: {str(e)}")


def main():
    """Main function to run feature importance analysis."""
    
    # Configuration
    BASE_DIR = "/Users/jerrylaivivemachi/DS PROJECT/J_DA_Project/Customer propensity to purchase dataset"
    MODELS_DIR = os.path.join(BASE_DIR, "ml_outputs")
    OUTPUT_DIR = os.path.join(BASE_DIR, "feature_analysis_outputs")
    
    # Validate paths
    if not os.path.exists(MODELS_DIR):
        logger.error(f"Models directory not found: {MODELS_DIR}")
        sys.exit(1)
    
    # Initialize and run analysis
    analyzer = FeatureImportanceAnalyzer(
        models_dir=MODELS_DIR,
        output_dir=OUTPUT_DIR
    )
    
    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis()


if __name__ == "__main__":
    main()