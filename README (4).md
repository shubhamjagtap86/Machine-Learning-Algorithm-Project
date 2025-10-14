


# 📊 Linear Regression on Advertising Dataset

This project demonstrates how to use linear regression to predict product sales based on advertising budgets across three media channels: TV, Radio, and Newspaper. It includes data loading, exploratory data analysis (EDA), feature selection, model training, and evaluation.

## 📁 Dataset

The dataset used is `Advertising.csv`, which contains 200 rows and the following columns:

- `TV`: Advertising budget spent on TV
- `Radio`: Advertising budget spent on Radio
- `Newspaper`: Advertising budget spent on Newspaper
- `Sales`: Sales generated

## 🧰 Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

## 📊 Exploratory Data Analysis (EDA)

- **Univariate Analysis**: Distribution plots for TV, Radio, and Newspaper budgets
- **Bivariate Analysis**: Relationship between each advertising channel and Sales
- **Multivariate Analysis**: Pairplot and correlation heatmap to understand feature interactions

## 🧹 Data Preprocessing

- Checked for missing values (none found)
- Dropped irrelevant column `Unnamed: 0`
- Outlier detection using boxplots
- Skipped scaling and transformation steps (not critical for this dataset)

## 📌 Feature Selection

- Selected `TV`, `Radio`, and `Newspaper` as input features
- Dropped `Unnamed: 0`
- Correlation analysis showed:
  - TV has the strongest correlation with Sales (0.78)
  - Radio has moderate correlation (0.57)
  - Newspaper has weak correlation (0.23)

## 🧪 Model Building

- Split data into training (80%) and testing (20%) sets
- Used `LinearRegression` from `sklearn`
- Trained the model on training data
- Predicted Sales on test data

## 📈 Model Evaluation

| Metric               | Value     |
|----------------------|-----------|
| R² Score             | 0.899     |
| Adjusted R²          | 0.891     |
| Mean Squared Error   | 3.17      |
| Root Mean Squared Error | 1.78  |
| Mean Absolute Error  | 1.46      |

## 🔍 Prediction Example

```python
# Predicting sales for TV=150, Radio=90, Newspaper=50
prediction = LR.predict([[150, 90, 50]])
# Output: 26.85
```

## 📌 Model Coefficients

```python
TV:        0.0447
Radio:     0.1892
Newspaper: 0.0028
Intercept: 2.9791
```

## 🚀 How to Run

1. Clone this repository
2. Ensure `Advertising.csv` is in the working directory
3. Run the notebook or script using Jupyter Notebook or any Python IDE

