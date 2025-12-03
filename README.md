 # Assignment4

XGBoost vs Neural Networks
EAS 510 – Assignment 4
**Author:** Raghulchellapandiyan Senthil Kumaran

---

## 1. Project Overview and Objectives

This project compares **XGBoost** with two **neural network regressors** (NN_v1 and NN_v2) for predicting **log–Airbnb price** across **12 cities** grouped into three tiers:

- **Large:** New York City, Los Angeles, San Francisco, Chicago
- **Medium:** Austin, Seattle, Denver, Portland
- **Small:** Asheville, Santa Cruz County, Salem-OR, Columbus

The notebook implements and documents:
1. **Automated data collection**
Scrapes the InsideAirbnb “Get the Data” web page.
- Automatically and correctly picks and finds the latest `listings.csv.gz` snapshot for each of the 12 cities.
- Downloads and extracts each city into a tier-specific folder.
`data/big/`, `data/medium/`, `data/small/`.

2. **City-wise preprocessing pipeline**
Clean price and text fields.
- Parses and converts using the function bathroom_text to numeric.
- Handles missing values and outliers.

Label encoding of the important categorical variables.
- Adds various features engineered to improve regressors.
- Conduct preliminary and subsequent evaluations of preprocessing through the use of boxplots and inspection reports.
- Conduct model and performance evaluation activities.

3. **Modeling and evaluation**  
   - Trains **per-city models**: XGBoost, NN_v1, NN_v2.  
   - Trains **tier-level models**: big-tier, medium-tier, small-tier.  
   - Performs and analyze **cross-tier neural network analysis**, training a composite NN for each tier and evaluating it on different tiers.  
   - Compares **RMSE**, **MAE**, and **R²** for all settings.
  
4. **Analysis conclusiones**

- Compares XGBoost vs NN_v1 vs NN_v2 per city.

- Compares and contrast Tier-Level Models.

- Performs analysis and comparison of cross-tier generalization behavior of NN_v1 model and NN_v2 model.

## 2. Instructions and ways to Run the Code
### Environment 2.1.

The notebook is designed for **Google Colab** but can run in any environment with:

Python 3.x
- `requests`, `beautifulsoup4`

These are: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`

- `xgboost`

- `tensorflow` / `keras`

In Colab these first cells install required packages:

```python

!pip install requests beautifulsoup4

!pip install xgboost shap tensorflow seaborn
```

### 2.2. Executing the Notebook

To get started, open the notebook file "raghulch_assignment4.ipynb" in either Jupyter or Google Colab. The notebook consists of four steps that you must follow in order.

1) The "Downloader" step creates and organize directories in the folder "data" (big, medium and small) and downloads the twelve city listings datasets (the *_listings.csv files) into these folders.

2) The "Preprocessing and Inspection" step contains the definition of the Preprocessing pipeline, the Preprocessing Inspection (before and after the Preprocessing pipeline is run), and the Boxplots that were generated.

3) The "Training" step contains the individual and particular city training and evaluates three models: XGBoost (XGB), NN_v1 and NN_v2. Additionally, this step also prints the RMSE, MAE and R² measurements.

4) The last and final step will be, "Tier-wise", combines all the cities that belong to the same tier and trains tier-based models. 
Cross-tier section: trains tier composite NNs and evaluates cross-tier predictions, generating tables and heatmaps.

No manual file editing is required; all paths are hard-coded to follow the data/<tier>/<city>_listings.csv naming convention.

## 3. Table of Data-Exact Months Utilized for Each City

The downloader always picks the latest snapshot available from InsideAirbnb at the time of execution. For documentation, the following table summarizes the representative months and approximate listing counts used, matching the assignment style:
| City               | Tier   | Data Snapshot Month | Listings Count |
|-------------------|--------|----------------------|----------------|
| New York City     | Big    | Oct 2025             | 36,111         |
| Los Angeles       | Big    | Sep 2025             | 45,886         |
| San Francisco     | Big    | Sep 2025             | 7,780          |
| Chicago           | Big    | May 2025             | 6,804          |
| Austin            | Medium | May 2025             | 15,187         |
| Seattle           | Medium | Sep 2025             | 6,295          |
| Denver            | Medium | Sep 2025             | 4,910          |
| Portland          | Medium | Sep 2025             | 4,425          |
| Asheville         | Small  | Jun 2025             | 2,876          |
| Santa Cruz County | Small  | Jun 2025             | 1,739          |
| Salem-OR          | Small  | Sep 2025             | 531            |
| Columbus          | Small  | Sep 2025             | 2,877          |

## 4. Summary and analysis of Preprocessing and Feature Engineering

### 4.1. Core Numeric Cleaning
Price cleaning
Removes $ and, from price and cast to float.

Drops rows with missing price.

Parsing of bathrooms is done (using bathrooms_text)

Retrieves and conversion numeric values (numbers basically) from each and every text in the data.

Missing values are imputed with the median.

Required Numeric number Features

Ensures existence and numeric type for:

Accommodates, bedrooms, beds, bathrooms_text

review_scores_*, number_of_reviews,

availability_365, minimum_nights, maximum_nights.

Fills missing values with median.

Review scores

Remaining missing values in individual review score fields are filled with column means.

### 4.2 Categorical Handling

room_type and neighbourhood_cleansed

Converted to strings and missing values replaced with "Unknown".

A grouped neighbourhood variable is created:

Top 20 most frequent neighborhoods kept as is.

All others grouped into "OTHER".

- room_type and neighbourhood_grouped are both label-encoded and stored as numeric values

### 4.3 Outlier Management using IQR Winsorization; 

The main variables (Price, Accommodates, Bedrooms, Beds, Bathrooms_Txt, Number_of_Reviews, Availability_365, Price_Per_Bedroom, Count_of_Amenities) will be calculated for Q1, Q3, and IQR = Q3 - Q1, and any value greater than [Q1 - (1.5 x IQR)] or less than [Q3 + (1.5 x IQR)] will be "Winsorized".

Reduces extreme spikes while keeping the majority of observations intact.

Before/after inspections

inspect_city_before() prints shape, head, missing values, numeric summaries, and category distributions.

inspect_city_after() prints shape and cleaned numeric statistics after preprocessing.

plot_boxplots_before_after() visualizes distributions for selected numeric columns to confirm that outliers have been smoothed and scales are more compact post-cleaning.

### 4.4. Engineered Features

At least four new features are explicitly engineered (the pipeline actually includes more):

price_per_bedroom = price / max(bedrooms, 1)

avg_review_score = mean of all 7 review score metrics

is_entire_home = indicator for room_type == "Entire home/apt"

amenities_count = approximate count of amenities from the amenities string

room_density = accommodates / (bedrooms + 1)

review_score_ratio = normalized score avg_review_score / 5.0

occupancy_estimate = availability_365 / (minimum_nights + 1)

#### The final feature vector contains:

All required numeric fields

Encoded categoricals (room_type_enc, neighbourhood_enc)

All engineered features above

The models learn to predict log(price) from this enriched, cleaned feature set.

## 5. Key Discoveries and findings in here - XGBoost vs. Neural Networks

### 5.1. Models by City Performance

The notebook trains on each of the twelve cities:

- XGBoost regressor (Boosted Trees)
- NN version 1: simple deep neural network with 2 hidden layers
- NN version 2: More layers with BatchNorm and Dropout

Notable Insights:

- XGBoost was consistently the best model for every city
- The model also produced the highest average R-squared value for all twelve cities.

Consistently lowest RMSE and MAE.

Performs extremely well in both big markets (NYC, LA, SF, Chicago) and smaller markets (Asheville, Columbus, etc.).

NN_v1 vs NN_v2

NN_v1 performs reasonably well in big cities where data volume is large.

NN_v1 performance degrades substantially in small cities and especially in Salem-OR, where R² becomes strongly negative.

NN_v2 improves stability and accuracy compared to NN_v1 in many cities, especially where data is moderately sized, but still does not surpass XGBoost.

All twelve cities show that XGBoost outperformed the two models based on neural network design in Total Model accuracy and performance stability through proactive model validation via several techniques.

### 5.2. Relative to their overall populations, the ranking of the cities is based on their relative population sizes. The cities are divided into three categories according to population size.

The cities categorized as part of the Top Tier are: Chicago, New York, San Francisco & Los Angeles.

The Medium Tier Cities are: Austin, Denver, Portland & Seattle.

The cities in the Small Tier are: Asheville, Columbus, Salem & Santa Cruz County.

Training data for XGBoost, NN_v1 & NN_v2 have been combined and organized into separate notebooks for each of the tiers.

#### Summary and conclusion of Results:

The XGBoost model had the highest R-squared score and lowest root-mean-square error (RMSE) score. Both NN_v1 and NN_v2 had the same score, but did not reach the XGBoost level of score.

In the Medium Tier, the XGBoost also provided the highest R-squared score again. The NN_v2 was slightly better than the NN_v1. NN_v1 & NN_v2 both fell significantly short of the XGBoost in terms of score.

In the Small Tier, the XGBoost model outperformed both NN models by considerable margins due to the relatively small and noisy data set available to both NN models.

#### Overall Conclusion by Tier:

The XGBoost model outperforms the totality of large, medium & small data. The NN models would receive improvement as they are provided with larger data sets, but they would still have a substantially lower predictive performance level than the XGBoost Model.

### 5.3. Cross-Tier Neural Network Generalization**


The purpose of this section is to investigate the extent to which a composite NN tier generalizes when a distribution shift occurs by creating composite NN models from scratch and then testing those same models across other tiers of NNs.
**Example or Instances to be mentioned:** The macro political tier NN (large) will be tested against both the medium and small tiers of NN's.
The macro political tier NN (medium) will be evaluated against both the macro political tier NN (large) and macro political tier NN (small).
The political small-tier NN is tested on the big-tier and medium-tier NN.

Overall, the NN_v1 and NN_v2 each have multiple metrics that are visualized in the form of heatmap and graph representations for comparison.

#### Major Findings:


Large Tier NNs provide a large amount of generalization to Medium Tier and Small Tier NNs.


NN_v2 also results in more accurate R² and RMSE values than NN_v1.

Medium → Big

NN_v1 collapses with strongly negative R².

NN_v2 maintains positives, strong R², which means much better generalization.
Medium → Small

Both NN_v1 and NN_v2 perform well; NN_v1 sometimes slightly better.

Medium-tier data transfers reasonably well to small markets.

Small → Big

Both NN_v1 and NN_v2 perform poorly (negative R²), but NN_v2 is notably less catastrophic than NN_v1.

Small-tier data is too limited and narrow to learn the structure of big-city pricing.

Small → Medium

Somewhat better than small → big, but still weaker than models trained on larger tiers.

#### Cross-tier conclusion:

NN_v2 is the better neural architecture for cross-tier generalization, especially when training on richer (big/medium) markets and testing on other tiers.

NN_v1 is fragile under domain shift, performing well only when train and test distributions are similar.

From a generalization viewpoint, richer tiers (big cities) produce models that transfer best; smaller tiers cannot scale upward.

## 6. Overall Conclusions

XGBoost vs Neural Networks (city and tier levels)

XGBoost is the overall winner:

Best R² and RMSE across all 12 cities.

Best performance on big, medium, and small tier composites.
Most robust and stable model in this assignment. Neural Network comparison (NN_v1 vs NN_v2) NN_v2, with BatchNorm and Dropout, is a clear improvement over NN_v1 in most cases: Better cross-tier generalization. Less prone to catastrophic failures under distribution shift. 

NN_v1 is competitive only in large data scenarios and fails badly in some small-tier settings. Preprocessing impact Consistent cleaning (price parsing, bathroom numeric extraction, missing value handling, IQR winsorization) and feature engineering (price_per_bedroom, avg_review_score, is_entire_home, amenities_count, density-style features) are crucial for stabilizing both tree-based and neural models. 

Before/after inspections and boxplots show that distributions become more balanced and amenable to learning after preprocessing. General lesson for tabular regression For rich, structured tabular data like Airbnb listings, XGBoost remains a very strong baseline. 

Neural networks can be improved with deeper architectures and regularization but still require careful design and large, diverse data to approach tree-based performance.
