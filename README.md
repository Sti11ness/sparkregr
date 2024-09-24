
# California Housing Price Prediction Project

## Project Overview

In this project, the goal is to build and evaluate linear regression models for predicting the median house value (`median_house_value`) in California based on housing data from 1990. The dataset contains several features that describe various aspects of housing blocks.

The dataset columns include:
- **longitude**: Longitude coordinate of the housing block.
- **latitude**: Latitude coordinate of the housing block.
- **housing_median_age**: The median age of houses in the block.
- **total_rooms**: The total number of rooms in all houses in the block.
- **total_bedrooms**: The total number of bedrooms in all houses in the block.
- **population**: The total population of the block.
- **households**: The number of households in the block.
- **median_income**: The median income of households in the block.
- **median_house_value**: The median house value of the block (target variable).
- **ocean_proximity**: The proximity of the block to the ocean.

The task is to predict the **median_house_value** using linear regression models and evaluate their performance on a test dataset. The following metrics are used to evaluate the model's performance:
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**
- **R-Squared (R²)**

## Models Built

Three different models of linear regression were built with varying treatment of the `ocean_proximity` categorical feature:
1. **Manual Encoding of `ocean_proximity` Feature**: Custom encoding of categorical values into numerical values.
2. **OneHotEncoding (OHE) of `ocean_proximity`**: Encoding the `ocean_proximity` categorical feature into multiple binary columns.
3. **Model without `ocean_proximity`**: Excluding the `ocean_proximity` feature entirely from the analysis.

## Project Pipeline

### 1. **Manual Encoding of `ocean_proximity` Feature**

1. **OceanProximityEncoder**: Custom encoding for `ocean_proximity`.
2. **Imputation**: Missing value imputation for numerical columns.
3. **Vector Assembler**: Combining all numerical features and the encoded categorical feature into one vector.
4. **Scaler**: Standardizing the features.
5. **Linear Regression**: Training the model using grid search for hyperparameter tuning.

### 2. **OneHotEncoding of `ocean_proximity` Feature**

1. **OHE Encoder**: Applying OneHotEncoding to the `ocean_proximity` feature.
2. **Imputation**: Handling missing values in numerical columns.
3. **Vector Assembler**: Combining numerical and OHE-transformed features.
4. **Scaler**: Standardizing the features.
5. **Linear Regression**: Training the model with grid search for tuning.

### 3. **Excluding `ocean_proximity` Feature**

1. **Imputation**: Handling missing values in numerical columns.
2. **Vector Assembler**: Combining all numerical features into one vector.
3. **Scaler**: Standardizing the features.
4. **Linear Regression**: Training the model with grid search for tuning.

## Model Evaluation

The models were evaluated using the following metrics:
- **RMSE**: Measures the average magnitude of the errors in prediction.
- **MAE**: Measures the average absolute difference between predicted and actual values.
- **R²**: Measures how well the model explains the variability of the target variable.

### Results:

| Model                          | RMSE         | MAE          | R²           |
|---------------------------------|--------------|--------------|--------------|
| Manual Encoding (`ocean_proximity`) | 66,652.47    | 48,495.29    | 0.6664       |
| OneHotEncoding (OHE)            | **66,000.33**| **47,814.54**| **0.6729**   |
| Without `ocean_proximity`       | 66,778.06    | 48,636.27    | 0.6651       |

### Conclusions:

- The **OneHotEncoding (OHE)** model performed the best in terms of all three metrics (RMSE, MAE, R²), suggesting that encoding the categorical feature using OHE was the most effective strategy.
- The **Manual Encoding** model had similar results but performed slightly worse than OHE.
- Excluding the `ocean_proximity` feature entirely led to the worst performance, highlighting the importance of including this feature in the model.

## Libraries and Tools Used

- **Apache Spark (PySpark)** for data processing and machine learning.
- **Matplotlib** and **Seaborn** for data visualization.
- **CrossValidator** and **GridSearch** for hyperparameter tuning.
  
## How to Run

1. Install required packages:
   ```bash
   pip install pyspark matplotlib seaborn
   ```

2. Load the dataset into a PySpark DataFrame.

3. Run the provided pipeline code to train the models and evaluate their performance.

4. Review the results and compare the models using the metrics provided.

## Conclusion

This project demonstrates the importance of properly handling categorical features in machine learning models. OneHotEncoding was the most effective strategy in this case, showing the highest accuracy across all evaluated metrics.
