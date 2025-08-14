# Training Dataset Exploration Findings

## Dataset Overview
- **Total Records:** 455,401 customer records
- **Features:** 25 columns (24 features + 1 target variable)
- **Missing Values:** None - complete dataset
- **Memory Usage:** 123.3 MB

## Target Variable Distribution
- **Variable:** `ordered` (binary classification)
- **Class Distribution:**
  - No Purchase (0): 436,308 records (95.81%)
  - Purchase (1): 19,093 records (4.19%)
- **Imbalance Ratio:** ~23:1 (highly imbalanced)

## Feature Categories

### User Interaction Features
- `basket_icon_click`: 9.92% engagement rate
- `basket_add_list`: 7.45% engagement rate
- `basket_add_detail`: 11.29% engagement rate
- `detail_wishlist_add`: 0.35% engagement rate
- `closed_minibasket_click`: 1.73% engagement rate
- `list_size_dropdown`: 23.04% engagement rate (highest interaction)

### Navigation & Discovery
- `sort_by`: 3.68% usage
- `image_picker`: 2.67% usage
- `promo_banner_click`: 1.62% click rate
- `account_page_click`: 0.36% engagement

### Checkout Process
- `sign_in`: 8.88% signed in users
- `saw_checkout`: 8.01% viewed checkout
- `checked_delivery_detail`: 6.29% checked delivery info
- `checked_returns_detail`: 0.92% checked returns

### Page Views
- `saw_homepage`: 29.00% (most viewed)
- `saw_delivery`: 0.55%
- `saw_sizecharts`: 0.04%
- `saw_account_upgrade`: 0.11%

### Device Distribution
- **Mobile:** 68.07% (dominant platform)
- **Computer:** 19.42%
- **Tablet:** 12.84%
- Note: Small overlap suggests multi-device usage

### User Segments
- **Returning Users:** 53.49%
- **New Users:** 46.51%
- **UK Location:** 93.32%
- **Non-UK:** 6.68%

## Key Insights

### Engagement Patterns
1. **Low Conversion Rate:** Only 4.19% of sessions result in purchases
2. **Mobile-First:** 68% of traffic comes from mobile devices
3. **High Browse Rate:** 29% see homepage but only 8% reach checkout
4. **List Browsing:** 23% use list size dropdown (browsing behavior)

### Modeling Implications
1. **Class Imbalance Challenge:** Severe imbalance requires special handling:
   - Consider SMOTE or other oversampling techniques
   - Use class weights in models
   - Focus on precision-recall metrics over accuracy
   - Consider ensemble methods or cost-sensitive learning

2. **Feature Engineering Opportunities:**
   - Combine basket interactions into engagement score
   - Create device preference indicators
   - Build funnel progression features (homepage → checkout)
   - Aggregate page view patterns

3. **Business Context:**
   - High mobile usage suggests mobile optimization importance
   - Low conversion typical for e-commerce (browse-heavy behavior)
   - UK-focused market with international presence

## Data Quality
- **Completeness:** 100% - no missing values
- **Data Types:** All behavioral features are binary (0/1)
- **UserID:** Anonymized unique identifiers
- **Memory Efficient:** Binary encoding keeps dataset compact

## Next Steps
1. Feature engineering to capture interaction patterns
2. Address class imbalance in modeling approach
3. Analyze feature correlations with purchase behavior
4. Segment analysis (device, location, user type)
5. Build predictive models with appropriate evaluation metrics