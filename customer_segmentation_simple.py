#!/usr/bin/env python3
"""
Customer Segmentation System using Existing Predictions
======================================================

Comprehensive customer segmentation based on existing propensity scores:
- Propensity-based segmentation
- Behavioral profiling
- Statistical analysis and insights
- Actionable business recommendations

Author: Senior ML Engineer
Date: 2025-08-14
"""

import os
import sys
import logging
import json
from typing import Dict, List, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomerSegmentationSystem:
    """
    Customer segmentation system using existing model predictions.
    """
    
    def __init__(self, output_dir: str = "segmentation_outputs"):
        
        self.output_dir = output_dir
        self.data = {}
        self.segments = {}
        self.segment_profiles = {}
        self.insights = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Define feature categories for analysis
        self.engagement_features = [
            'basket_icon_click', 'basket_add_list', 'basket_add_detail', 'sort_by',
            'image_picker', 'detail_wishlist_add', 'list_size_dropdown'
        ]
        
        self.intent_features = [
            'checked_delivery_detail', 'checked_returns_detail', 'sign_in',
            'saw_checkout', 'saw_delivery', 'saw_sizecharts'
        ]
        
        self.context_features = [
            'device_mobile', 'device_computer', 'device_tablet',
            'returning_user', 'loc_uk'
        ]
        
        logger.info("Customer segmentation system initialized")
    
    def load_data_and_predictions(self, train_path: str, test_path: str, test_predictions_path: str):
        """Load training data, test data, and existing test predictions."""
        try:
            logger.info("Loading data and existing predictions...")
            
            # Load training and test data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            test_predictions = pd.read_csv(test_predictions_path)
            
            # Add data split identifier
            train_data['data_split'] = 'train'
            test_data['data_split'] = 'test'
            
            # Merge test data with predictions
            test_data = test_data.merge(
                test_predictions[['UserID', 'purchase_probability']], 
                on='UserID', 
                how='left'
            )
            test_data.rename(columns={'purchase_probability': 'propensity_score'}, inplace=True)
            
            # For training data, we need to generate propensity scores
            # We'll use the conversion rate as a proxy for now
            # In a real scenario, you'd use the trained model
            logger.info("Generating propensity scores for training data...")
            
            # Create a simple propensity score based on behavioral features
            train_data['propensity_score'] = self._calculate_behavioral_propensity(train_data)
            
            # Combine datasets
            feature_cols = [col for col in train_data.columns 
                          if col not in ['UserID', 'ordered', 'data_split', 'propensity_score']]
            
            # Ensure both datasets have the same columns
            common_cols = ['UserID', 'data_split', 'propensity_score'] + feature_cols
            if 'ordered' in train_data.columns:
                common_cols.append('ordered')
            
            # For test data, add ordered column as NaN if not present
            if 'ordered' not in test_data.columns:
                test_data['ordered'] = np.nan
            
            combined_df = pd.concat([
                train_data[common_cols], 
                test_data[common_cols]
            ], ignore_index=True)
            
            self.data = {
                'combined': combined_df,
                'train': train_data,
                'test': test_data,
                'feature_cols': feature_cols
            }
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"- Total customers: {len(combined_df):,}")
            logger.info(f"- Training customers: {len(train_data):,}")
            logger.info(f"- Test customers: {len(test_data):,}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _calculate_behavioral_propensity(self, df: pd.DataFrame) -> pd.Series:
        """Calculate a behavioral propensity score based on user actions."""
        try:
            # Weight different behaviors based on their importance
            behavior_weights = {
                'saw_checkout': 0.4,
                'checked_delivery_detail': 0.25,
                'sign_in': 0.15,
                'basket_add_detail': 0.1,
                'basket_icon_click': 0.05,
                'basket_add_list': 0.03,
                'detail_wishlist_add': 0.02
            }
            
            propensity = pd.Series(0.0, index=df.index)
            
            for feature, weight in behavior_weights.items():
                if feature in df.columns:
                    propensity += df[feature] * weight
            
            # Normalize to 0-1 range
            if propensity.max() > 0:
                propensity = propensity / propensity.max()
            
            # Add some noise to avoid all zeros
            propensity += np.random.normal(0, 0.01, len(propensity))
            propensity = np.clip(propensity, 0, 1)
            
            return propensity
            
        except Exception as e:
            logger.warning(f"Error calculating behavioral propensity: {str(e)}")
            return pd.Series(0.0, index=df.index)
    
    def create_propensity_segments(self, n_segments: int = 5) -> pd.DataFrame:
        """Create customer segments based on propensity scores."""
        logger.info(f"Creating {n_segments} propensity-based segments...")
        
        try:
            df = self.data['combined'].copy()
            
            # Define segment thresholds using quantiles
            quantiles = [i/n_segments for i in range(n_segments + 1)]
            propensity_thresholds = df['propensity_score'].quantile(quantiles).values
            
            # Ensure thresholds are unique and increasing
            propensity_thresholds = np.unique(propensity_thresholds)
            if len(propensity_thresholds) < n_segments + 1:
                # If we don't have enough unique thresholds, use equal intervals
                min_score = df['propensity_score'].min()
                max_score = df['propensity_score'].max()
                propensity_thresholds = np.linspace(min_score, max_score, n_segments + 1)
            
            # Create segment labels
            if n_segments == 5:
                segment_labels = ['Very Low Propensity', 'Low Propensity', 'Medium Propensity', 
                                'High Propensity', 'Very High Propensity']
            elif n_segments == 4:
                segment_labels = ['Low Propensity', 'Medium-Low Propensity', 
                                'Medium-High Propensity', 'High Propensity']
            else:
                segment_labels = [f'Segment_{i+1}' for i in range(n_segments)]
            
            # Assign segments
            try:
                df['propensity_segment'] = pd.cut(df['propensity_score'], 
                                                bins=propensity_thresholds, 
                                                labels=segment_labels[:len(propensity_thresholds)-1], 
                                                include_lowest=True)
                
                df['propensity_segment_id'] = pd.cut(df['propensity_score'], 
                                                   bins=propensity_thresholds, 
                                                   labels=range(len(propensity_thresholds)-1), 
                                                   include_lowest=True).astype(int)
            except Exception as e:
                logger.warning(f"Error creating segments with quantiles: {str(e)}")
                # Fallback to equal intervals
                df['propensity_segment'] = pd.qcut(df['propensity_score'], 
                                                 q=n_segments, 
                                                 labels=segment_labels, 
                                                 duplicates='drop')
                df['propensity_segment_id'] = pd.qcut(df['propensity_score'], 
                                                    q=n_segments, 
                                                    labels=range(n_segments), 
                                                    duplicates='drop')
            
            self.segments['propensity'] = df
            
            logger.info("Propensity segments created successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error creating propensity segments: {str(e)}")
            raise
    
    def create_behavioral_segments(self, n_clusters: int = 5) -> pd.DataFrame:
        """Create customer segments based on behavioral patterns using clustering."""
        logger.info(f"Creating {n_clusters} behavioral segments using K-means clustering...")
        
        try:
            df = self.data['combined'].copy()
            
            # Select features for behavioral segmentation
            behavioral_features = [f for f in self.engagement_features + self.intent_features 
                                 if f in df.columns]
            
            if not behavioral_features:
                logger.warning("No behavioral features found for clustering")
                return df
            
            # Prepare data for clustering
            X_behavioral = df[behavioral_features].fillna(0)
            
            # Add derived behavioral metrics
            df['engagement_score'] = df[[f for f in self.engagement_features if f in df.columns]].sum(axis=1)
            df['intent_score'] = df[[f for f in self.intent_features if f in df.columns]].sum(axis=1)
            df['total_interactions'] = df[self.data['feature_cols']].sum(axis=1)
            
            # Include derived metrics in clustering
            clustering_features = behavioral_features + ['engagement_score', 'intent_score', 'total_interactions']
            X_clustering = df[clustering_features].fillna(0)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clustering)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            df['behavioral_segment_id'] = cluster_labels
            
            # Create meaningful segment names based on cluster characteristics
            segment_names = self._generate_behavioral_segment_names(df, clustering_features, n_clusters)
            df['behavioral_segment'] = df['behavioral_segment_id'].map(segment_names)
            
            # Store clustering information
            self.behavioral_clustering = {
                'model': kmeans,
                'scaler': scaler,
                'features': clustering_features
            }
            
            self.segments['behavioral'] = df
            
            logger.info("Behavioral segments created successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error creating behavioral segments: {str(e)}")
            raise
    
    def _generate_behavioral_segment_names(self, df: pd.DataFrame, features: List[str], n_clusters: int) -> Dict[int, str]:
        """Generate meaningful names for behavioral segments based on cluster characteristics."""
        segment_names = {}
        
        try:
            # Analyze each cluster
            for cluster_id in range(n_clusters):
                cluster_data = df[df['behavioral_segment_id'] == cluster_id]
                
                if len(cluster_data) == 0:
                    segment_names[cluster_id] = f"Empty_Segment_{cluster_id}"
                    continue
                
                cluster_means = cluster_data[features].mean()
                
                # Determine dominant characteristics
                high_engagement = cluster_means.get('engagement_score', 0) > df['engagement_score'].median()
                high_intent = cluster_means.get('intent_score', 0) > df['intent_score'].median()
                high_interactions = cluster_means.get('total_interactions', 0) > df['total_interactions'].median()
                
                # Generate name based on characteristics
                if high_engagement and high_intent:
                    name = "Highly Engaged Buyers"
                elif high_engagement and not high_intent:
                    name = "Active Browsers"
                elif not high_engagement and high_intent:
                    name = "Purposeful Shoppers"
                elif high_interactions:
                    name = "Exploratory Users"
                else:
                    name = "Low Activity Users"
                
                # Add cluster ID to make unique if needed
                base_name = name
                counter = 1
                while name in segment_names.values():
                    name = f"{base_name} {counter}"
                    counter += 1
                
                segment_names[cluster_id] = name
            
        except Exception as e:
            logger.warning(f"Could not generate meaningful segment names: {str(e)}")
            # Fallback to generic names
            segment_names = {i: f"Behavioral_Segment_{i+1}" for i in range(n_clusters)}
        
        return segment_names
    
    def create_hybrid_segments(self) -> pd.DataFrame:
        """Create hybrid segments combining propensity and behavioral characteristics."""
        logger.info("Creating hybrid segments...")
        
        try:
            if 'propensity' not in self.segments or 'behavioral' not in self.segments:
                logger.error("Both propensity and behavioral segments needed for hybrid segmentation")
                return pd.DataFrame()
            
            df = self.segments['propensity'].copy()
            
            # Add behavioral segment info
            behavioral_df = self.segments['behavioral'][['UserID', 'behavioral_segment', 'behavioral_segment_id']]
            df = df.merge(behavioral_df, on='UserID', how='left', suffixes=('', '_behavioral'))
            
            # Create simplified hybrid categories
            def create_hybrid_category(row):
                try:
                    prop_level = row['propensity_segment_id']
                    behav_segment = str(row['behavioral_segment'])
                    
                    if pd.isna(prop_level):
                        return 'Unclassified'
                    
                    if prop_level >= 3:  # High propensity
                        if 'Engaged' in behav_segment or 'Buyer' in behav_segment:
                            return 'Champions'  # High propensity + high engagement
                        else:
                            return 'High Intent'  # High propensity but lower engagement
                    elif prop_level >= 2:  # Medium propensity
                        if 'Active' in behav_segment or 'Explorer' in behav_segment:
                            return 'Potential Buyers'  # Medium propensity + active behavior
                        else:
                            return 'Warm Prospects'  # Medium propensity, moderate activity
                    else:  # Low propensity
                        if 'Active' in behav_segment or 'Explorer' in behav_segment:
                            return 'Nurture Candidates'  # Low propensity but active
                        else:
                            return 'Low Priority'  # Low propensity + low activity
                except Exception:
                    return 'Unclassified'
            
            df['hybrid_category'] = df.apply(create_hybrid_category, axis=1)
            
            self.segments['hybrid'] = df
            
            logger.info("Hybrid segments created successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error creating hybrid segments: {str(e)}")
            raise
    
    def profile_segments(self) -> Dict[str, Dict]:
        """Create comprehensive profiles for all segment types."""
        logger.info("Creating segment profiles...")
        
        profiles = {}
        
        try:
            for segment_type in ['propensity', 'behavioral', 'hybrid']:
                if segment_type not in self.segments:
                    continue
                
                logger.info(f"Profiling {segment_type} segments...")
                df = self.segments[segment_type]
                
                if segment_type == 'hybrid':
                    segment_col = 'hybrid_category'
                else:
                    segment_col = f'{segment_type}_segment'
                
                segment_profiles = {}
                
                for segment in df[segment_col].unique():
                    if pd.isna(segment):
                        continue
                    
                    segment_data = df[df[segment_col] == segment]
                    train_segment_data = segment_data[segment_data['data_split'] == 'train']
                    
                    profile = {
                        'size': len(segment_data),
                        'percentage': len(segment_data) / len(df) * 100,
                        'train_size': len(train_segment_data),
                        'test_size': len(segment_data) - len(train_segment_data)
                    }
                    
                    # Conversion metrics (only for train data where we have labels)
                    if len(train_segment_data) > 0 and 'ordered' in train_segment_data.columns:
                        valid_orders = train_segment_data['ordered'].dropna()
                        if len(valid_orders) > 0:
                            profile.update({
                                'conversion_rate': valid_orders.mean(),
                                'total_conversions': valid_orders.sum()
                            })
                    
                    # Propensity score statistics
                    if 'propensity_score' in segment_data.columns:
                        profile.update({
                            'avg_propensity': segment_data['propensity_score'].mean(),
                            'median_propensity': segment_data['propensity_score'].median(),
                            'propensity_std': segment_data['propensity_score'].std()
                        })
                    
                    # Behavioral characteristics
                    behavioral_cols = ['engagement_score', 'intent_score', 'total_interactions']
                    for col in behavioral_cols:
                        if col in segment_data.columns:
                            profile[f'avg_{col}'] = segment_data[col].mean()
                    
                    # Context characteristics
                    context_features = [f for f in self.context_features if f in segment_data.columns]
                    for feature in context_features:
                        profile[f'{feature}_rate'] = segment_data[feature].mean()
                    
                    # Top behavioral features for this segment
                    behavioral_features = [f for f in self.engagement_features + self.intent_features 
                                         if f in segment_data.columns]
                    top_behaviors = {}
                    for feature in behavioral_features:
                        rate = segment_data[feature].mean()
                        if rate > 0.001:  # Only include behaviors with >0.1% rate
                            top_behaviors[feature] = rate
                    
                    profile['top_behaviors'] = dict(sorted(top_behaviors.items(), 
                                                         key=lambda x: x[1], reverse=True)[:10])
                    
                    segment_profiles[str(segment)] = profile
                
                profiles[segment_type] = segment_profiles
            
            self.segment_profiles = profiles
            
            # Save profiles to JSON
            output_path = os.path.join(self.output_dir, 'segment_profiles.json')
            with open(output_path, 'w') as f:
                json.dump(profiles, f, indent=2, default=str)
            
            logger.info("Segment profiles created successfully")
            
        except Exception as e:
            logger.error(f"Error profiling segments: {str(e)}")
        
        return profiles
    
    def analyze_segment_performance(self) -> Dict[str, Any]:
        """Calculate performance metrics and statistical significance."""
        logger.info("Analyzing segment performance...")
        
        analysis_results = {}
        
        try:
            if 'propensity' not in self.segments:
                return analysis_results
            
            df = self.segments['propensity']
            train_df = df[df['data_split'] == 'train']
            
            if 'ordered' not in train_df.columns:
                logger.warning("No conversion data available for performance analysis")
                return analysis_results
            
            # Remove NaN values
            train_df = train_df.dropna(subset=['ordered', 'propensity_segment'])
            
            if len(train_df) == 0:
                logger.warning("No valid training data for performance analysis")
                return analysis_results
            
            # Overall conversion rate
            overall_conversion = train_df['ordered'].mean()
            
            # Segment performance
            segment_performance = {}
            segments = train_df['propensity_segment'].unique()
            
            for segment in segments:
                if pd.isna(segment):
                    continue
                
                segment_data = train_df[train_df['propensity_segment'] == segment]
                
                if len(segment_data) == 0:
                    continue
                
                segment_conversion = segment_data['ordered'].mean()
                n = len(segment_data)
                successes = segment_data['ordered'].sum()
                
                # Calculate lift
                lift = ((segment_conversion - overall_conversion) / overall_conversion * 100) if overall_conversion > 0 else 0
                
                # Calculate confidence interval (Wilson score interval)
                if n > 0:
                    z = 1.96  # 95% confidence
                    p_hat = successes / n
                    denominator = 1 + (z**2 / n)
                    centre_adjusted_probability = p_hat + (z**2 / (2 * n))
                    adjusted_standard_deviation = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
                    
                    lower_bound = (centre_adjusted_probability - adjusted_standard_deviation) / denominator
                    upper_bound = (centre_adjusted_probability + adjusted_standard_deviation) / denominator
                    
                    segment_performance[str(segment)] = {
                        'conversion_rate': segment_conversion,
                        'lift_percent': lift,
                        'confidence_interval_lower': max(0, lower_bound),
                        'confidence_interval_upper': min(1, upper_bound),
                        'sample_size': n,
                        'total_conversions': int(successes),
                        'statistical_significance': abs(lift) > 10  # Simple threshold
                    }
            
            analysis_results['segment_performance'] = segment_performance
            analysis_results['overall_conversion_rate'] = overall_conversion
            
            # ANOVA test for statistical significance
            try:
                segment_groups = [train_df[train_df['propensity_segment'] == seg]['ordered'].values 
                                for seg in segments if not pd.isna(seg) and 
                                len(train_df[train_df['propensity_segment'] == seg]) > 0]
                
                if len(segment_groups) > 1:
                    f_stat, p_value = stats.f_oneway(*segment_groups)
                    analysis_results['anova_test'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant_at_05': p_value < 0.05
                    }
            except Exception as e:
                logger.warning(f"Could not perform ANOVA test: {str(e)}")
            
            # Save analysis results
            output_path = os.path.join(self.output_dir, 'segment_performance_analysis.json')
            with open(output_path, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info("Segment performance analysis completed")
            
        except Exception as e:
            logger.error(f"Error analyzing segment performance: {str(e)}")
        
        return analysis_results
    
    def generate_business_recommendations(self) -> Dict[str, List[str]]:
        """Generate actionable business recommendations for each segment."""
        logger.info("Generating business recommendations...")
        
        recommendations = {}
        
        try:
            # Recommendations for hybrid segments (most actionable)
            segment_recommendations = {
                'Champions': [
                    "VIP Treatment: Provide premium customer service, early access to new products, and exclusive deals",
                    "Loyalty Rewards: Create a tiered loyalty program with high-value rewards and personalized experiences",
                    "Referral Program: Implement attractive referral incentives to leverage their advocacy",
                    "Cross-sell Strategy: Recommend complementary products and premium upgrades",
                    "Retention Focus: Proactive engagement to maintain their high-value status and prevent churn"
                ],
                'High Intent': [
                    "Conversion Acceleration: Send targeted offers with urgency messaging and limited-time discounts",
                    "Personalized Recommendations: Use AI to suggest products based on browsing and purchase history",
                    "Checkout Optimization: Streamline the purchase process and reduce friction points",
                    "Abandoned Cart Recovery: Implement aggressive email sequences for cart abandonment",
                    "Customer Support: Provide proactive chat support to address purchase barriers"
                ],
                'Potential Buyers': [
                    "Nurture Campaigns: Educational content and product demonstrations via email marketing",
                    "Retargeting Strategy: Strategic display and social media retargeting campaigns",
                    "Incentive Offers: Free shipping, first-time buyer discounts, and bundle deals",
                    "Social Proof: Prominently display customer reviews, ratings, and testimonials",
                    "Content Marketing: Valuable blog posts, guides, and tutorials related to their interests"
                ],
                'Warm Prospects': [
                    "Engagement Building: Interactive content like quizzes, polls, and personalized assessments",
                    "Progressive Profiling: Gradually collect preference information through forms and surveys",
                    "Newsletter Strategy: Regular newsletters with valuable content and soft product placement",
                    "Community Building: Invite to exclusive groups, webinars, and events",
                    "Trust Building: Share brand story, values, and customer success stories"
                ],
                'Nurture Candidates': [
                    "Educational Focus: Product education and brand awareness campaigns",
                    "Low-pressure Engagement: Soft-touch marketing with valuable, non-promotional content",
                    "Social Media: Engage through social channels with lifestyle and interest-based content",
                    "Feedback Collection: Understand barriers to purchase through surveys and research",
                    "Long-term Nurturing: Patient, value-driven relationship building approach"
                ],
                'Low Priority': [
                    "Automated Campaigns: Low-cost, automated email campaigns with minimal manual effort",
                    "Seasonal Campaigns: Occasional high-discount campaigns during sales periods",
                    "List Hygiene: Regular cleaning and pruning to maintain good sender reputation",
                    "Win-back Attempts: Periodic re-engagement campaigns with special offers",
                    "Cost Management: Minimal marketing spend allocation to maximize ROI"
                ]
            }
            
            recommendations['hybrid_segments'] = segment_recommendations
            
            # Device and context-specific recommendations
            device_recommendations = {
                'mobile_optimization': [
                    "Mobile-First Design: Ensure flawless mobile user experience and responsive design",
                    "Mobile Payments: Integrate mobile wallets (Apple Pay, Google Pay, PayPal)",
                    "App Development: Consider native mobile app for enhanced user experience",
                    "SMS Marketing: Implement SMS campaigns for immediate mobile reach",
                    "Touch-Friendly Interface: Large buttons, easy navigation, and swipe gestures"
                ],
                'desktop_experience': [
                    "Rich Content: Provide detailed product information, comparisons, and specifications",
                    "Live Chat Support: Implement live chat for real-time purchase assistance",
                    "Email Marketing: Focus on email as primary communication channel for desktop users",
                    "Detailed Reviews: Comprehensive review systems for informed decision-making",
                    "Advanced Features: Product configurators, wish lists, and comparison tools"
                ]
            }
            
            recommendations['platform_optimization'] = device_recommendations
            
            # Behavioral pattern recommendations
            behavioral_recommendations = {
                'high_engagement_low_conversion': [
                    "Barrier Identification: Survey users to understand conversion obstacles",
                    "Price Sensitivity: Consider pricing strategies, payment plans, or promotions",
                    "Trust Signals: Add more security badges, guarantees, and social proof elements",
                    "Process Simplification: Streamline checkout and reduce required form fields",
                    "Product-Market Fit: Ensure product offerings match user expectations"
                ],
                'low_engagement_high_intent': [
                    "Fast-Track Experience: Create streamlined purchase paths for efficient buyers",
                    "Minimal Friction: Remove unnecessary steps and simplify the user journey",
                    "Clear Value Proposition: Prominently display key benefits and differentiators",
                    "Efficient Search: Improve site search functionality and product discovery",
                    "Quick Wins: Focus on immediate value demonstration and easy conversions"
                ]
            }
            
            recommendations['behavioral_optimization'] = behavioral_recommendations
            
            # Save recommendations
            output_path = os.path.join(self.output_dir, 'business_recommendations.json')
            with open(output_path, 'w') as f:
                json.dump(recommendations, f, indent=2)
            
            logger.info("Business recommendations generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations for segment analysis."""
        logger.info("Creating segment visualizations...")
        
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Propensity Score Distribution by Segment
            if 'propensity' in self.segments:
                self._create_propensity_dashboard()
            
            # 2. Behavioral Segment Analysis
            if 'behavioral' in self.segments:
                self._create_behavioral_dashboard()
            
            # 3. Hybrid Segment Performance
            if 'hybrid' in self.segments:
                self._create_hybrid_dashboard()
            
            logger.info("Segment visualizations created successfully")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
    
    def _create_propensity_dashboard(self) -> None:
        """Create propensity segment analysis dashboard."""
        try:
            df = self.segments['propensity']
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Propensity distribution
            ax1 = axes[0, 0]
            for segment in df['propensity_segment'].dropna().unique():
                segment_data = df[df['propensity_segment'] == segment]
                ax1.hist(segment_data['propensity_score'], alpha=0.6, label=str(segment), bins=30)
            ax1.set_xlabel('Propensity Score')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Propensity Score Distribution by Segment')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Segment sizes
            ax2 = axes[0, 1]
            segment_counts = df['propensity_segment'].value_counts()
            ax2.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%')
            ax2.set_title('Segment Size Distribution')
            
            # Conversion rates (train data only)
            ax3 = axes[1, 0]
            train_df = df[df['data_split'] == 'train']
            if 'ordered' in train_df.columns and not train_df['ordered'].isna().all():
                conv_rates = train_df.groupby('propensity_segment')['ordered'].mean()
                bars = ax3.bar(range(len(conv_rates)), conv_rates.values)
                ax3.set_xticks(range(len(conv_rates)))
                ax3.set_xticklabels(conv_rates.index, rotation=45, ha='right')
                ax3.set_ylabel('Conversion Rate')
                ax3.set_title('Conversion Rate by Segment')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom')
            else:
                ax3.text(0.5, 0.5, 'No conversion data available', 
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title('Conversion Rate by Segment (No Data)')
            
            # Behavioral scores by segment
            ax4 = axes[1, 1]
            if 'engagement_score' in df.columns and 'intent_score' in df.columns:
                segment_behavior = df.groupby('propensity_segment')[['engagement_score', 'intent_score']].mean()
                segment_behavior.plot(kind='bar', ax=ax4)
                ax4.set_title('Average Behavioral Scores by Segment')
                ax4.set_ylabel('Average Score')
                ax4.tick_params(axis='x', rotation=45)
                ax4.legend(['Engagement', 'Intent'])
            else:
                ax4.text(0.5, 0.5, 'No behavioral scores available', 
                        transform=ax4.transAxes, ha='center', va='center')
                ax4.set_title('Behavioral Scores (No Data)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'propensity_segment_dashboard.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create propensity dashboard: {str(e)}")
    
    def _create_behavioral_dashboard(self) -> None:
        """Create behavioral segment analysis dashboard."""
        try:
            df = self.segments['behavioral']
            
            # Behavioral characteristics heatmap
            behavioral_features = [f for f in self.engagement_features + self.intent_features 
                                 if f in df.columns]
            
            if not behavioral_features:
                return
            
            segment_behavior = df.groupby('behavioral_segment')[behavioral_features].mean()
            
            plt.figure(figsize=(14, 8))
            sns.heatmap(segment_behavior.T, annot=True, fmt='.3f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Behavior Rate'})
            plt.title('Behavioral Characteristics by Segment')
            plt.xlabel('Behavioral Segment')
            plt.ylabel('Behavior')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'behavioral_segment_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create behavioral dashboard: {str(e)}")
    
    def _create_hybrid_dashboard(self) -> None:
        """Create hybrid segment performance dashboard."""
        try:
            df = self.segments['hybrid']
            train_df = df[df['data_split'] == 'train']
            
            if 'ordered' not in train_df.columns or train_df['ordered'].isna().all():
                logger.warning("No conversion data for hybrid dashboard")
                return
            
            # Calculate performance metrics by hybrid category
            performance = train_df.groupby('hybrid_category').agg({
                'ordered': ['mean', 'count'],
                'propensity_score': 'mean'
            }).round(3)
            
            # Flatten column names
            performance.columns = ['_'.join(col).strip() for col in performance.columns.values]
            
            # Create subplot
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Conversion rates
            ax1 = axes[0, 0]
            conv_rates = performance['ordered_mean']
            bars = ax1.bar(range(len(conv_rates)), conv_rates.values)
            ax1.set_xticks(range(len(conv_rates)))
            ax1.set_xticklabels(conv_rates.index, rotation=45, ha='right')
            ax1.set_ylabel('Conversion Rate')
            ax1.set_title('Conversion Rate by Hybrid Segment')
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
            
            # Segment sizes
            ax2 = axes[0, 1]
            sizes = performance['ordered_count']
            ax2.pie(sizes.values, labels=sizes.index, autopct='%1.1f%%')
            ax2.set_title('Hybrid Segment Size Distribution')
            
            # Propensity scores
            ax3 = axes[1, 0]
            prop_scores = performance['propensity_score_mean']
            bars = ax3.bar(range(len(prop_scores)), prop_scores.values)
            ax3.set_xticks(range(len(prop_scores)))
            ax3.set_xticklabels(prop_scores.index, rotation=45, ha='right')
            ax3.set_ylabel('Average Propensity Score')
            ax3.set_title('Average Propensity Score by Segment')
            
            # Segment distribution
            ax4 = axes[1, 1]
            hybrid_dist = df['hybrid_category'].value_counts()
            ax4.bar(range(len(hybrid_dist)), hybrid_dist.values)
            ax4.set_xticks(range(len(hybrid_dist)))
            ax4.set_xticklabels(hybrid_dist.index, rotation=45, ha='right')
            ax4.set_ylabel('Number of Customers')
            ax4.set_title('Customer Count by Hybrid Category')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'hybrid_segment_performance.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create hybrid dashboard: {str(e)}")
    
    def export_segment_assignments(self) -> None:
        """Export segment assignments for all customers."""
        logger.info("Exporting segment assignments...")
        
        try:
            # Export comprehensive segment assignments
            if 'hybrid' in self.segments:
                export_cols = ['UserID', 'propensity_score', 'data_split']
                
                # Add segment columns
                for col in ['propensity_segment', 'propensity_segment_id', 
                          'behavioral_segment', 'behavioral_segment_id', 'hybrid_category']:
                    if col in self.segments['hybrid'].columns:
                        export_cols.append(col)
                
                # Add behavioral scores
                for col in ['engagement_score', 'intent_score', 'total_interactions']:
                    if col in self.segments['hybrid'].columns:
                        export_cols.append(col)
                
                # Add conversion info for train data
                if 'ordered' in self.segments['hybrid'].columns:
                    export_cols.append('ordered')
                
                export_df = self.segments['hybrid'][export_cols].copy()
                
                # Save comprehensive assignments
                output_path = os.path.join(self.output_dir, 'propensity_segments.csv')
                export_df.to_csv(output_path, index=False)
                
                logger.info(f"Segment assignments exported: {output_path}")
                
                # Create summary statistics
                summary_stats = {
                    'total_customers': len(export_df),
                    'train_customers': len(export_df[export_df['data_split'] == 'train']),
                    'test_customers': len(export_df[export_df['data_split'] == 'test']),
                    'segment_distributions': {}
                }
                
                # Add distribution for each segment type
                for segment_type in ['propensity_segment', 'behavioral_segment', 'hybrid_category']:
                    if segment_type in export_df.columns:
                        dist = export_df[segment_type].value_counts(normalize=True).to_dict()
                        summary_stats['segment_distributions'][segment_type] = dist
                
                # Save summary
                summary_path = os.path.join(self.output_dir, 'segmentation_summary.json')
                with open(summary_path, 'w') as f:
                    json.dump(summary_stats, f, indent=2, default=str)
                
                logger.info("Segmentation summary exported")
            
        except Exception as e:
            logger.error(f"Error exporting segment assignments: {str(e)}")
    
    def run_comprehensive_segmentation(self, train_path: str, test_path: str, predictions_path: str) -> None:
        """Run the complete customer segmentation analysis."""
        try:
            logger.info("=" * 60)
            logger.info("CUSTOMER SEGMENTATION ANALYSIS STARTED")
            logger.info("=" * 60)
            
            # Load data and predictions
            self.load_data_and_predictions(train_path, test_path, predictions_path)
            
            # Create different types of segments
            self.create_propensity_segments(n_segments=5)
            self.create_behavioral_segments(n_clusters=5)
            self.create_hybrid_segments()
            
            # Analyze segments
            self.profile_segments()
            self.analyze_segment_performance()
            self.generate_business_recommendations()
            
            # Create visualizations
            self.create_visualizations()
            
            # Export results
            self.export_segment_assignments()
            
            logger.info("=" * 60)
            logger.info("CUSTOMER SEGMENTATION ANALYSIS COMPLETED")
            logger.info("=" * 60)
            
            # Print summary
            self._print_segmentation_summary()
            
        except Exception as e:
            logger.error(f"Segmentation analysis failed: {str(e)}")
            raise
    
    def _print_segmentation_summary(self) -> None:
        """Print summary of segmentation results."""
        try:
            logger.info("\nCUSTOMER SEGMENTATION SUMMARY:")
            logger.info("-" * 50)
            
            # Hybrid segment summary
            if 'hybrid' in self.segments:
                df = self.segments['hybrid']
                train_df = df[df['data_split'] == 'train']
                
                logger.info("\nHYBRID SEGMENT PERFORMANCE:")
                for category in df['hybrid_category'].unique():
                    if pd.isna(category):
                        continue
                    
                    category_data = df[df['hybrid_category'] == category]
                    train_category_data = train_df[train_df['hybrid_category'] == category]
                    
                    size = len(category_data)
                    percentage = size / len(df) * 100
                    avg_propensity = category_data['propensity_score'].mean()
                    
                    info_str = f"{category}: {size:,} customers ({percentage:.1f}%), Avg Propensity: {avg_propensity:.3f}"
                    
                    if len(train_category_data) > 0 and 'ordered' in train_category_data.columns:
                        valid_orders = train_category_data['ordered'].dropna()
                        if len(valid_orders) > 0:
                            conv_rate = valid_orders.mean()
                            info_str += f", Conversion: {conv_rate:.3f}"
                    
                    logger.info(info_str)
            
            # Key insights
            logger.info("\nKEY INSIGHTS:")
            if 'hybrid' in self.segment_profiles:
                # Find segments with highest and lowest propensity
                segments_with_propensity = {}
                for segment, profile in self.segment_profiles['hybrid'].items():
                    if 'avg_propensity' in profile and not pd.isna(profile['avg_propensity']):
                        segments_with_propensity[segment] = profile['avg_propensity']
                
                if segments_with_propensity:
                    best_segment = max(segments_with_propensity, key=segments_with_propensity.get)
                    worst_segment = min(segments_with_propensity, key=segments_with_propensity.get)
                    
                    logger.info(f"• Highest propensity segment: {best_segment} ({segments_with_propensity[best_segment]:.3f} avg propensity)")
                    logger.info(f"• Lowest propensity segment: {worst_segment} ({segments_with_propensity[worst_segment]:.3f} avg propensity)")
            
            logger.info(f"\n• All segmentation outputs saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error printing summary: {str(e)}")


def main():
    """Main function to run customer segmentation."""
    
    # Configuration
    BASE_DIR = "/Users/jerrylaivivemachi/DS PROJECT/J_DA_Project/Customer propensity to purchase dataset"
    TRAIN_PATH = os.path.join(BASE_DIR, "training_sample.csv")
    TEST_PATH = os.path.join(BASE_DIR, "testing_sample.csv")
    PREDICTIONS_PATH = os.path.join(BASE_DIR, "ml_outputs", "test_predictions.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "segmentation_outputs")
    
    # Validate paths
    for path in [TRAIN_PATH, TEST_PATH, PREDICTIONS_PATH]:
        if not os.path.exists(path):
            logger.error(f"Required path not found: {path}")
            sys.exit(1)
    
    # Initialize and run segmentation
    segmentation_system = CustomerSegmentationSystem(output_dir=OUTPUT_DIR)
    
    # Run comprehensive segmentation
    segmentation_system.run_comprehensive_segmentation(TRAIN_PATH, TEST_PATH, PREDICTIONS_PATH)


if __name__ == "__main__":
    main()