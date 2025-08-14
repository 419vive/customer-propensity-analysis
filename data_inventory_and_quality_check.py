"""
Data Inventory and Quality Check Script
========================================
This script performs comprehensive data quality checks and inventory analysis
on the customer propensity to purchase datasets.

Author: Jerry Lai
Date: 2025-01-14
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_datasets():
    """Load training and testing datasets"""
    print("=" * 70)
    print("LOADING DATASETS")
    print("=" * 70)
    
    train_df = pd.read_csv('training_sample.csv')
    test_df = pd.read_csv('testing_sample.csv')
    
    print(f"✓ Training data loaded: {train_df.shape[0]:,} rows x {train_df.shape[1]} columns")
    print(f"✓ Testing data loaded: {test_df.shape[0]:,} rows x {test_df.shape[1]} columns")
    
    return train_df, test_df

def data_inventory(df, dataset_name):
    """Perform comprehensive data inventory"""
    print(f"\n{'=' * 70}")
    print(f"DATA INVENTORY - {dataset_name.upper()}")
    print("=" * 70)
    
    # Basic information
    print("\n📊 DATASET DIMENSIONS:")
    print(f"   • Rows: {df.shape[0]:,}")
    print(f"   • Columns: {df.shape[1]}")
    print(f"   • Total cells: {df.shape[0] * df.shape[1]:,}")
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\n💾 MEMORY USAGE:")
    print(f"   • Total: {memory_usage:.2f} MB")
    print(f"   • Average per row: {(memory_usage * 1024 / df.shape[0]):.2f} KB")
    
    # Data types distribution
    print(f"\n🔤 DATA TYPES DISTRIBUTION:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   • {dtype}: {count} columns ({count/len(df.columns)*100:.1f}%)")
    
    # Column inventory
    print(f"\n📝 COLUMN INVENTORY:")
    print("   Column Name                    | Type    | Unique Values | Sample Values")
    print("   " + "-" * 65)
    
    for col in df.columns:
        unique_count = df[col].nunique()
        sample_values = df[col].value_counts().head(3).index.tolist()
        sample_str = str(sample_values[:2])[1:-1] if len(sample_values) > 2 else str(sample_values)[1:-1]
        print(f"   {col:30} | {str(df[col].dtype):7} | {unique_count:13,} | {sample_str[:25]}")
    
    return dtype_counts

def quality_checks(df, dataset_name):
    """Perform comprehensive data quality checks"""
    print(f"\n{'=' * 70}")
    print(f"DATA QUALITY CHECKS - {dataset_name.upper()}")
    print("=" * 70)
    
    quality_issues = []
    
    # 1. Missing Values Check
    print("\n🔍 MISSING VALUES CHECK:")
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing == 0:
        print("   ✅ No missing values detected - Dataset is complete!")
    else:
        print(f"   ⚠️ Total missing values: {total_missing:,} ({total_missing/(df.shape[0]*df.shape[1])*100:.2f}%)")
        missing_cols = missing_counts[missing_counts > 0]
        for col, count in missing_cols.items():
            percentage = (count / len(df)) * 100
            print(f"      • {col}: {count:,} ({percentage:.2f}%)")
            quality_issues.append(f"Missing values in {col}")
    
    # 2. Duplicate Records Check
    print("\n🔍 DUPLICATE RECORDS CHECK:")
    duplicates = df.duplicated().sum()
    
    if duplicates == 0:
        print("   ✅ No duplicate records found!")
    else:
        print(f"   ⚠️ Found {duplicates:,} duplicate records ({duplicates/len(df)*100:.2f}%)")
        quality_issues.append(f"{duplicates} duplicate records")
    
    # 3. UserID Uniqueness Check
    if 'UserID' in df.columns:
        print("\n🔍 USERID UNIQUENESS CHECK:")
        duplicate_users = df['UserID'].duplicated().sum()
        
        if duplicate_users == 0:
            print("   ✅ All UserIDs are unique!")
        else:
            print(f"   ⚠️ Found {duplicate_users:,} duplicate UserIDs")
            quality_issues.append(f"{duplicate_users} duplicate UserIDs")
    
    # 4. Binary Features Validation
    print("\n🔍 BINARY FEATURES VALIDATION:")
    binary_cols = [col for col in df.columns if col not in ['UserID']]
    invalid_binary = []
    
    for col in binary_cols:
        unique_values = df[col].unique()
        if not set(unique_values).issubset({0, 1, np.nan}):
            invalid_binary.append(col)
    
    if not invalid_binary:
        print("   ✅ All binary features contain only valid values (0 or 1)!")
    else:
        print(f"   ⚠️ Found {len(invalid_binary)} columns with invalid binary values:")
        for col in invalid_binary:
            print(f"      • {col}: {df[col].unique()}")
            quality_issues.append(f"Invalid binary values in {col}")
    
    # 5. Data Consistency Checks
    print("\n🔍 DATA CONSISTENCY CHECKS:")
    consistency_issues = []
    
    # Check device columns sum
    if all(col in df.columns for col in ['device_mobile', 'device_computer', 'device_tablet']):
        device_sum = df[['device_mobile', 'device_computer', 'device_tablet']].sum(axis=1)
        invalid_device = (device_sum == 0).sum()
        
        if invalid_device == 0:
            print("   ✅ Device type consistency: All records have at least one device type")
        else:
            print(f"   ⚠️ Device inconsistency: {invalid_device:,} records with no device type")
            consistency_issues.append(f"{invalid_device} records with no device type")
    
    # Check if users who ordered must have seen checkout
    if 'ordered' in df.columns and 'saw_checkout' in df.columns:
        ordered_without_checkout = ((df['ordered'] == 1) & (df['saw_checkout'] == 0)).sum()
        
        if ordered_without_checkout == 0:
            print("   ✅ Logical consistency: All orders went through checkout")
        else:
            print(f"   ⚠️ Logical inconsistency: {ordered_without_checkout:,} orders without checkout view")
            consistency_issues.append(f"{ordered_without_checkout} orders without checkout")
    
    # Check basket actions consistency
    if 'basket_add_detail' in df.columns and 'basket_add_list' in df.columns:
        total_basket_adds = df[['basket_add_detail', 'basket_add_list']].max(axis=1)
        basket_without_icon = ((total_basket_adds == 1) & (df.get('basket_icon_click', 0) == 0)).sum()
        
        if basket_without_icon > 0:
            print(f"   ⚠️ Basket inconsistency: {basket_without_icon:,} basket adds without icon click")
            consistency_issues.append(f"{basket_without_icon} basket adds without icon click")
    
    if consistency_issues:
        quality_issues.extend(consistency_issues)
    
    # 6. Outlier Detection for Aggregated Features
    print("\n🔍 FEATURE ENGAGEMENT ANALYSIS:")
    
    # Calculate total interactions per user
    interaction_cols = [col for col in df.columns if col not in ['UserID', 'ordered', 'loc_uk', 
                                                                  'returning_user', 'device_mobile', 
                                                                  'device_computer', 'device_tablet']]
    
    if interaction_cols:
        total_interactions = df[interaction_cols].sum(axis=1)
        
        print(f"   • Mean interactions per session: {total_interactions.mean():.2f}")
        print(f"   • Median interactions per session: {total_interactions.median():.0f}")
        print(f"   • Max interactions per session: {total_interactions.max():.0f}")
        
        # Identify highly engaged users
        high_engagement_threshold = total_interactions.quantile(0.95)
        high_engaged_users = (total_interactions > high_engagement_threshold).sum()
        print(f"   • Highly engaged sessions (>95th percentile): {high_engaged_users:,} ({high_engaged_users/len(df)*100:.2f}%)")
    
    return quality_issues

def statistical_summary(df, dataset_name):
    """Generate statistical summary for numeric columns"""
    print(f"\n{'=' * 70}")
    print(f"STATISTICAL SUMMARY - {dataset_name.upper()}")
    print("=" * 70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if 'UserID' in df.columns:
        numeric_cols = [col for col in numeric_cols if col != 'UserID']
    
    if numeric_cols:
        print("\n📈 ENGAGEMENT METRICS (% of users):")
        engagement_stats = pd.DataFrame({
            'Feature': numeric_cols,
            'Engagement_Rate': [f"{df[col].mean()*100:.2f}%" for col in numeric_cols],
            'User_Count': [f"{df[col].sum():,}" for col in numeric_cols]
        })
        
        engagement_stats = engagement_stats.sort_values('Engagement_Rate', ascending=False)
        
        print("\n   Top 10 Most Used Features:")
        for idx, row in engagement_stats.head(10).iterrows():
            print(f"   {idx+1:2}. {row['Feature']:30} | {row['Engagement_Rate']:>7} | {row['User_Count']:>8} users")
        
        print("\n   Bottom 5 Least Used Features:")
        for idx, row in engagement_stats.tail(5).iterrows():
            print(f"   {idx-4:2}. {row['Feature']:30} | {row['Engagement_Rate']:>7} | {row['User_Count']:>8} users")
    
    # Target variable analysis
    if 'ordered' in df.columns:
        print("\n🎯 TARGET VARIABLE ANALYSIS:")
        conversion_rate = df['ordered'].mean() * 100
        print(f"   • Conversion rate: {conversion_rate:.2f}%")
        print(f"   • Purchases: {df['ordered'].sum():,}")
        print(f"   • Non-purchases: {(df['ordered'] == 0).sum():,}")
        print(f"   • Class imbalance ratio: 1:{int((df['ordered'] == 0).sum() / df['ordered'].sum())}")

def compare_datasets(train_df, test_df):
    """Compare training and testing datasets for consistency"""
    print(f"\n{'=' * 70}")
    print("DATASET COMPARISON - TRAINING vs TESTING")
    print("=" * 70)
    
    # Column comparison
    print("\n🔄 COLUMN COMPARISON:")
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    
    if train_cols == test_cols:
        print("   ✅ Columns match perfectly between datasets!")
    else:
        print("   ⚠️ Column mismatch detected:")
        
        if train_cols - test_cols:
            print(f"   • Columns only in training: {train_cols - test_cols}")
        if test_cols - train_cols:
            print(f"   • Columns only in testing: {test_cols - train_cols}")
    
    # Feature distribution comparison
    print("\n📊 FEATURE DISTRIBUTION COMPARISON:")
    common_cols = list(train_cols.intersection(test_cols))
    
    if 'UserID' in common_cols:
        common_cols.remove('UserID')
    
    distribution_diffs = []
    
    for col in common_cols[:10]:  # Check first 10 columns for brevity
        train_mean = train_df[col].mean()
        test_mean = test_df[col].mean()
        
        if train_mean > 0:
            diff_percent = abs(train_mean - test_mean) / train_mean * 100
            
            if diff_percent > 10:  # Flag if difference > 10%
                distribution_diffs.append((col, train_mean, test_mean, diff_percent))
    
    if not distribution_diffs:
        print("   ✅ Feature distributions are consistent between datasets!")
    else:
        print("   ⚠️ Significant distribution differences found:")
        for col, train_mean, test_mean, diff in distribution_diffs:
            print(f"   • {col}: Train={train_mean:.3f}, Test={test_mean:.3f} ({diff:.1f}% difference)")
    
    # Size comparison
    print("\n📏 SIZE COMPARISON:")
    print(f"   • Training: {train_df.shape[0]:,} rows")
    print(f"   • Testing: {test_df.shape[0]:,} rows")
    print(f"   • Ratio: 1:{test_df.shape[0]/train_df.shape[0]:.2f}")

def generate_quality_report(train_issues, test_issues):
    """Generate final quality report"""
    print(f"\n{'=' * 70}")
    print("FINAL DATA QUALITY REPORT")
    print("=" * 70)
    
    total_issues = len(train_issues) + len(test_issues)
    
    if total_issues == 0:
        print("\n🎉 EXCELLENT DATA QUALITY!")
        print("   No critical issues found in either dataset.")
        print("   The data is ready for modeling.")
    else:
        print(f"\n⚠️ TOTAL ISSUES FOUND: {total_issues}")
        
        if train_issues:
            print("\n   Training Dataset Issues:")
            for issue in train_issues:
                print(f"   • {issue}")
        
        if test_issues:
            print("\n   Testing Dataset Issues:")
            for issue in test_issues:
                print(f"   • {issue}")
    
    print("\n📋 RECOMMENDATIONS:")
    print("   1. Address class imbalance using SMOTE, undersampling, or class weights")
    print("   2. Consider feature engineering to capture interaction patterns")
    print("   3. Implement cross-validation to ensure model generalization")
    print("   4. Monitor for data drift in production deployment")
    print("   5. Use stratified sampling to maintain class distribution")

def main():
    """Main execution function"""
    print("=" * 70)
    print(" DATA INVENTORY AND QUALITY CHECK SYSTEM")
    print(" Customer Propensity to Purchase Analysis")
    print(f" Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Load datasets
    train_df, test_df = load_datasets()
    
    # Perform inventory for both datasets
    train_dtypes = data_inventory(train_df, "Training Dataset")
    test_dtypes = data_inventory(test_df, "Testing Dataset")
    
    # Perform quality checks
    train_issues = quality_checks(train_df, "Training Dataset")
    test_issues = quality_checks(test_df, "Testing Dataset")
    
    # Statistical summaries
    statistical_summary(train_df, "Training Dataset")
    statistical_summary(test_df, "Testing Dataset")
    
    # Compare datasets
    compare_datasets(train_df, test_df)
    
    # Generate final report
    generate_quality_report(train_issues, test_issues)
    
    print("\n" + "=" * 70)
    print("✓ Data inventory and quality check completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()